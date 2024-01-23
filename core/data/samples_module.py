import os
import shutil
import subprocess
import torch as th
import lightning as L
from core.extract import extract_world
from core.data.samples_dataset import WorldSampleDataset
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
        batch_size: int = 8,
        num_workers: int = 8,
        data_dir: str | os.PathLike = "outputs",
        raw_data_dir: str | os.PathLike = "raw_outputs",
        intermediate_data_dir: str | os.PathLike = "intermediate_outputs",
        device: th.device = th.device("cpu"),
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
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

        raise NotImplementedError("Implement me!")

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
    
    def setup(self: Self, stage: str | None = None) -> None:
        getattr(self, f"setup_{stage}", lambda: None)()

    def setup_fit(self: Self) -> None:
        self.train_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "train"),
            self.sample_size,
            device = self.device,
        )

        self.val_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "val"),
            self.sample_size,
            device = self.device,
        )

    def setup_test(self: Self) -> None:
        self.test_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "test"),
            self.sample_size,
            device = self.device,
        )

    def setup_predict(self: Self) -> None:
        self.predict_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "predict"),
            self.sample_size,
            device = self.device,
        )

    def train_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
