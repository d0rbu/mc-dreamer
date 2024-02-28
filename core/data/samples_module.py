import os
import shutil
import torch as th
import lightning as L
from core.extract import extract_world
from core.data.samples_dataset import WorldSampleDataset, WorldSampleDatasetMode
from torch.utils.data import DataLoader
from typing import Self


class WorldSampleDataModule(L.LightningDataModule):
    """
    Data module for world sample datasets.

    For raw data dir, intermediate data dir, and output dir, we have the functions:
    - scrape_data: scrape data from the internet -> raw data dir
    - extract_data: extract data from raw data dir -> intermediate data dir
    - convert_data: convert data from intermediate data dir -> output dir
    """

    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        tube_length: int | None = 8,
        dataset_mode: WorldSampleDatasetMode = WorldSampleDatasetMode.NORMAL,
        batch_size: int = 8,
        num_workers: int = 8,
        data_dir: str | os.PathLike = "outputs",
        raw_data_dir: str | os.PathLike = "raw_outputs",
        intermediate_data_dir: str | os.PathLike = "intermediate_outputs",
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.tube_length = tube_length
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.intermediate_data_dir = intermediate_data_dir
        self.device = device

    def scrape_data(
        self: Self,
        raw_data_dir: str | os.PathLike | None = None,
    ) -> None:
        if raw_data_dir is None:
            raw_data_dir = self.raw_data_dir

        raise NotImplementedError("Scraping data from data module not implemented")

    def extract_data(
        self: Self,
        raw_data_dir: str | os.PathLike | None = None,
        intermediate_dir: str | os.PathLike | None = None,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        if raw_data_dir is None:
            raw_data_dir = self.raw_data_dir

        if intermediate_dir is None:
            intermediate_dir = self.intermediate_data_dir

        if output_dir is None:
            output_dir = self.data_dir

        for data_path in os.listdir(raw_data_dir):
            extract_world(
                data_path,
                self.data_dir,
                self.intermediate_data_dir,
            )

        # Clear intermediate_data_dir
        shutil.rmtree(self.intermediate_data_dir, ignore_errors=True)
        os.makedirs(self.intermediate_data_dir, exist_ok=True)

        # Copy output_dir to intermediate_data_dir
        for data_path in os.listdir(output_dir):
            shutil.copy(
                os.path.join(output_dir, data_path),
                os.path.join(intermediate_dir, data_path),
            )

        # Clear output_dir
        shutil.rmtree(self.data_dir, ignore_errors=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def convert_data(
        self: Self,
        intermediate_data_dir: str | os.PathLike | None = None,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        if intermediate_data_dir is None:
            intermediate_data_dir = self.intermediate_data_dir

        if output_dir is None:
            output_dir = self.data_dir

        raise NotImplementedError("Implement me!")

    def prepare_data(self: Self) -> None:
        if len(os.listdir(self.data_dir)) > 0:
            return
        elif len(os.listdir(self.intermediate_data_dir)) == 0:
            self.convert_data()
        elif len(os.listdir(self.raw_data_dir)) == 0:
            self.extract_data()
        else:
            self.scrape_data()

        self.prepare_data()

    def setup(self: Self, stage: str) -> None:
        split_mappings = {
            "fit": ["train", "val"],
            "test": ["test"],
            "predict": ["predict"],
        }

        for split in split_mappings[stage]:
            setattr(self, f"{split}_dataset", WorldSampleDataset(
                data_dir = self.data_dir,
                split = split,
                sample_size = self.sample_size,
                tube_length = self.tube_length,
                dataset_mode = self.dataset_mode,
                device = self.device,
            ))

            # sets self.train_dataloader, self.val_dataloader, self.test_dataloader, etc.
            setattr(self, f"{split}_dataloader", DataLoader(
                getattr(self, f"{split}_dataset"),
                batch_size = self.batch_size,
                shuffle = split == "train",
                num_workers = self.num_workers,
            ))
