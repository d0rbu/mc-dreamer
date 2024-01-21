import os
import json
import pydantic
import torch as th
from torch.utils.data import Dataset
from typing import Self
from functools import cache


class FileMetadata(pydantic.BaseModel):
    name: str
    num_samples: int


class DatasetMetadata(pydantic.BaseModel):
    total_samples: int
    files: list[FileMetadata]


class WorldSampleDataset(Dataset):
    def __init__(
        self: Self,
        data_dir: str | os.PathLike,
        split: str = "train",
        sample_size: tuple[int, int, int] = (16, 16, 16),
        metadata_file: str | os.PathLike = "metadata.json",
        transform: th.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        self.sample_size = sample_size
        self.transform = transform

        assert os.path.isdir(self.data_dir), f"{self.data_dir} is not a directory"

        # Confirm that all files in data_dir are of valid sample size
        sample_size_string = "x".join(map(str, sample_size))
        for file in os.listdir(data_dir):
            if file.startswith(sample_size_string) or file == metadata_file:
                continue
            else:
                raise ValueError(
                    f"Invalid sample size for file {file}, expected {sample_size}"
                )

        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        assert os.path.isfile(metadata_path), f"{metadata_path} is not a file"
        with open(metadata_path) as metadata_file:
            self.metadata = json.load(metadata_file)
        self.metadata = DatasetMetadata(**self.metadata)

        # Load into memory
        self.data = []
        self.volume_indices = []
        self.sample_sizes = []
        for file_metadata in self.metadata.files:
            file_path = os.path.join(data_dir, file_metadata.name)
            data = th.load(file_path)
            self.data.append(data)
            self.volume_indices.append(th.nonzero(data))
            self.sample_sizes.append(file_metadata.num_samples)
        
        assert len(self.data) == len(self.volume_indices) == len(self.sample_sizes)
        assert sum(self.sample_sizes) == self.metadata.total_samples, (
            f"Total samples in metadata ({self.metadata.total_samples}) "
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
        return self.metadata.total_samples

    def __getitem__(self, index: int) -> th.Tensor:
        file_index, sample_index = self._get_data_indices(index)
        sample = self.data[file_index][sample_index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
