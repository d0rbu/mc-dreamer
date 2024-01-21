import os
import shutil
import subprocess
import lightning as L
from core.extract import extract_world
from core.data.samples_dataset import WorldSampleDataset
from torch.utils.data import DataLoader
from typing import Self


class WorldSampleDataModule(L.LightningDataModule):
    def __init__(
        self: Self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        batch_size: int = 8,
        num_workers: int = 8,
        data_dir: str | os.PathLike = "outputs",
        raw_data_dir: str | os.PathLike = "raw_outputs",
        intermediate_data_dir: str | os.PathLike = "intermediate_outputs",
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.intermediate_data_dir = intermediate_data_dir

    def scrape_data(
        self: Self,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        if output_dir is None:
            output_dir = self.raw_data_dir

        raise NotImplementedError("Implement me!")

    def extract_data(
        self: Self,
        input_dir: str | os.PathLike | None = None,
        intermediate_dir: str | os.PathLike | None = None,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        if input_dir is None:
            input_dir = self.raw_data_dir

        if intermediate_dir is None:
            intermediate_dir = self.intermediate_data_dir

        if output_dir is None:
            output_dir = self.data_dir

        for data_path in os.listdir(input_dir):
            extract_world(
                data_path,
                self.data_dir,
                self.intermediate_data_dir,
            )

        # Clear intermediate_data_dir
        shutil.rmtree(self.intermediate_data_dir, ignore_errors=True)
        os.makedirs(self.intermediate_data_dir, exist_ok=True)

    def prepare_data(self: Self) -> None:
        if len(os.listdir(self.data_dir)) > 0:
            return

        if len(os.listdir(self.raw_data_dir)) == 0:
            self.scrape_data()

        self.extract_data()
    
    def setup(self: Self, stage: str | None = None) -> None:
        getattr(self, f"setup_{stage}", lambda: None)()

    def setup_fit(self: Self) -> None:
        self.train_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "train"),
            self.sample_size,
        )

        self.val_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "val"),
            self.sample_size,
        )

    def setup_test(self: Self) -> None:
        self.test_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "test"),
            self.sample_size,
        )

    def setup_predict(self: Self) -> None:
        self.predict_dataset = WorldSampleDataset(
            os.path.join(self.data_dir, "predict"),
            self.sample_size,
        )

    def train_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self: Self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
