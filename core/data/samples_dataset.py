import os
import json
import torch as th
from torch.utils.data import Dataset
from pydantic import BaseModel
from typing import Self
from functools import cache


class FileMetadata(BaseModel):
    path: str


class DatasetMetadata(BaseModel):
    score_threshold: float
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    train: list[FileMetadata]
    val: list[FileMetadata]
    test: list[FileMetadata]


class WorldSampleDataset(Dataset):
    def __init__(
        self: Self,
        data_dir: str | os.PathLike,
        split: str = "train",
        sample_size: tuple[int, int, int] = (16, 16, 16),
        metadata_file: str | os.PathLike = "metadata.json",
        transform: th.nn.Module | None = None,
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.transform = transform
        self.device = device

        assert os.path.isdir(data_dir), f"{data_dir} is not a directory"

        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        assert os.path.isfile(metadata_path), f"{metadata_path} is not a file"
        with open(metadata_path) as metadata_file:
            self.metadata = json.load(metadata_file)
        self.metadata = DatasetMetadata(**self.metadata)

        # Confirm that all files in are of valid sample size
        sample_size_string = "x".join(map(str, sample_size))
        self.files = [file for file in self.metadata[split] if file.path.startswith(sample_size_string)]
        self.num_samples = self.metadata[f"num_{split}_samples"]

        # Load into memory
        self.data = []
        self.volume_indices = []
        self.sample_sizes = []
        for file_metadata in self.files:
            file_path = os.path.join(data_dir, file_metadata.path)
            raw_data = th.load(file_path, map_location=self.device)
            region_data = raw_data["region"]
            score_data = raw_data["scores"]
            assert region_data.shape == score_data.shape, (
                f"Region data shape {region_data.shape} does not match score data shape {score_data.shape}"
            )

            score_mask = score_data > self.metadata.score_threshold
            if not th.any(score_mask):
                continue    # Skip file if no scores are above threshold

            self.data.append(region_data)
            self.volume_indices.append(th.nonzero(score_mask).astype(th.uint16))
            self.sample_sizes.append(th.sum(score_mask).item())

        assert len(self.data) == len(self.volume_indices) == len(self.sample_sizes)
        assert sum(self.sample_sizes) == self.num_samples, (
            f"Total samples in {split} set in metadata ({self.num_samples}) "
            f"does not match sum of sample sizes ({sum(self.sample_sizes)})"
        )

    @cache
    def _get_data_indices(self: Self, sample_index: int) -> tuple[int, int]:
        # Return index of file and index of sample within file
        for i, sample_size in enumerate(self.sample_sizes):
            if sample_index < sample_size:
                return (i, sample_index)
            else:
                sample_index -= sample_size

    def _get_sample(self: Self, data_index: int, sample_index: int) -> th.Tensor:
        volume_index = self.volume_indices[data_index][sample_index]
        data = self.data[data_index]
        sample_slice = tuple(
            slice(
                volume_index[i],
                volume_index[i] + self.sample_size[i],
            )
            for i in range(3)
        )

        return data[sample_slice]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> th.Tensor:
        file_index, sample_index = self._get_data_indices(index)
        sample = self._get_sample(file_index, sample_index)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
