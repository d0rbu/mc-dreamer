import os
import json
import torch as th
from torch.utils.data import Dataset
from pydantic import BaseModel
from typing import Self, Callable, Sequence
from enum import StrEnum
from functools import cache


class FileMetadata(BaseModel):
    path: str
    size: int


class DatasetMetadata(BaseModel):
    score_threshold: float
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    train: list[FileMetadata]
    val: list[FileMetadata]
    test: list[FileMetadata]


class WorldSampleDatasetMode(StrEnum):
    NORMAL = "normal"
    STRUCTURE = "structure"
    COLOR = "color"


class WorldSampleDataset(Dataset):
    def __init__(
        self: Self,
        data_dir: str | os.PathLike,
        split: str = "train",
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int | None = 8,
        dataset_mode: WorldSampleDatasetMode = WorldSampleDatasetMode.NORMAL,
        metadata_file: str | os.PathLike = "metadata.json",
        transform: Callable[[th.Tensor], th.Tensor] | None = None,
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.sample_y_slice_size = (1, sample_size[1], sample_size[2])
        self.tube_length = tube_length
        self.dataset_mode = dataset_mode
        self.transform: Callable[[th.Tensor], th.Tensor] = transform if transform is not None else lambda x: x
        self.device = device

        assert os.path.isdir(data_dir), f"{data_dir} is not a directory"

        # load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        assert os.path.isfile(metadata_path), f"{metadata_path} is not a file"
        with open(metadata_path) as metadata_file:
            self.metadata = json.load(metadata_file)
        self.metadata = DatasetMetadata(**self.metadata)

        # confirm that all files in are of valid sample size
        sample_size_string = "x".join(map(str, sample_size))
        split_files = getattr(self.metadata, split)
        self.files = [file for file in split_files if os.path.basename(file.path).startswith(sample_size_string)]
        self.num_samples = getattr(self.metadata, f"num_{split}_samples")

        # load into memory
        self.data = []
        self.volume_indices = []
        self.volume_index_sets = []
        self.sample_sizes = []
        for file_metadata in self.files:
            file_path = os.path.join(data_dir, file_metadata.path)
            assert os.path.isfile(file_path), f"{file_path} is not a file"

            if file_metadata.size <= 0:
                continue

            raw_data = th.load(file_path, map_location=self.device)
            region_data = th.permute(raw_data["region"], (1, 2, 0))  # (X, Y, Z) -> (Y, Z, X)
            score_data = th.permute(raw_data["scores"], (1, 2, 0))  # (X, Y, Z) -> (Y, Z, X)

            expected_score_data_shape = (region_data.shape[0] + sample_size[0] - 1, region_data.shape[1] + sample_size[1] - 1, region_data.shape[2] + sample_size[2] - 1)

            assert expected_score_data_shape == score_data.shape, (
                f"Expected score data shape {expected_score_data_shape} does not match score data shape {score_data.shape}"
            )

            score_mask = score_data > self.metadata.score_threshold

            assert score_mask.sum() == file_metadata.size, (
                f"Score mask sum {score_mask.sum()} does not match file size {file_metadata.size}"
            )

            if not th.any(score_mask):
                continue    # skip file if no scores are above threshold

            high_score_indices = th.nonzero(score_mask).to(th.int16)

            self.data.append(region_data)
            self.volume_indices.append(high_score_indices)
            self.volume_index_sets.append(set(map(tuple, high_score_indices.tolist())))
            self.sample_sizes.append(th.sum(score_mask).item())

        assert len(self.data) == len(self.volume_indices) == len(self.volume_index_sets) == len(self.sample_sizes)
        assert sum(self.sample_sizes) == self.num_samples, (
            f"Total samples in {split} set in metadata ({self.num_samples}) "
            f"does not match sum of sample sizes ({sum(self.sample_sizes)})"
        )

        print(f"Loaded {len(self.data)} files for {split} set with {self.num_samples} samples")

    @cache
    def _get_data_indices(self: Self, sample_index: int) -> tuple[int, int]:
        # return index of file and index of sample within file
        for i, sample_size in enumerate(self.sample_sizes):
            if sample_index < sample_size:
                return (i, sample_index)
            else:
                sample_index -= sample_size
    
    def _get_slice(
        self: Self,
        volume_index: tuple[int, int, int],
        sample_size: tuple[int, int, int] | None = None
    ) -> tuple[slice, slice, slice]:
        if sample_size is None:
            sample_size = self.sample_size

        return tuple(
            slice(
                volume_index[i],
                volume_index[i] + self.sample_size[i]
            )
            for i in range(3)
        )

    def _get_sample(self: Self, data_index: int, sample_index: int) -> tuple[th.Tensor, int, th.Tensor | None, th.Tensor | None]:
        volume_index = self.volume_indices[data_index][sample_index]
        data = self.data[data_index]
        sample_slice = self._get_slice(volume_index)

        if self.dataset_mode != WorldSampleDatasetMode.STRUCTURE:
            return self.transform(data[sample_slice]), volume_index[0].item(), None, None

        volume_below_index = (volume_index[0] - 1, volume_index[1], volume_index[2])
        volume_above_index = (volume_index[0] + 1, volume_index[1], volume_index[2])

        if self.tube_length and volume_below_index in self.volume_index_sets[data_index]:
            below_sample_slice = self._get_slice(volume_below_index, sample_size=self.sample_y_slice_size)
            previous_tube = self.transform(data[below_sample_slice]).flatten()[-self.tube_length:]
        else:
            previous_tube = None

        if self.tube_length and volume_above_index in self.volume_index_sets[data_index]:
            above_sample_slice = self._get_slice(volume_above_index, sample_size=self.sample_y_slice_size)
            next_tube = self.transform(data[above_sample_slice]).flatten()[:self.tube_length]
        else:
            next_tube = None

        return self.transform(data[sample_slice]), volume_index[0].item(), previous_tube, next_tube

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[th.Tensor, int]:
        file_index, sample_index = self._get_data_indices(index)
        sample, y_index, previous_tube, next_tube = self._get_sample(file_index, sample_index)

        if self.dataset_mode == WorldSampleDatasetMode.NORMAL:
            return sample
        elif self.dataset_mode == WorldSampleDatasetMode.STRUCTURE:
            structure = sample > 0
            prev_tube_structure = previous_tube > 0 if previous_tube is not None else None
            next_tube_structure = next_tube > 0 if next_tube is not None else None

            return structure, y_index, prev_tube_structure, next_tube_structure
        elif self.dataset_mode == WorldSampleDatasetMode.COLOR:
            return {
                "sample": sample,
                "control": {
                    "structure": sample > 0,
                    "y_index": y_index,
                },
            }
