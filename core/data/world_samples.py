import os
import subprocess
import lightning as L
from core.extract import extract_world
from torch.utils.data import DataLoader
from typing import Self


class WorldSampleDataModule(L.LightningDataModule):
    def __init__(
        self,
        sample_size: tuple[int, int, int] = (16, 16, 16),
        data_dir: str | os.PathLike = "outputs",
        raw_data_dir: str | os.PathLike = "raw_outputs",
        intermediate_data_dir: str | os.PathLike = "intermediate_outputs",
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.intermediate_data_dir = intermediate_data_dir

    def scrape_data(self) -> None:
        # Use self.raw_data_dir
        raise NotImplementedError("Implement me!")

    def extract_data(
        self: Self,
        data_paths: list[str | os.PathLike],
    ) -> None:
        for data_path in data_paths:
            extract_world(
                data_path,
                self.data_dir,
                self.intermediate_data_dir,
            )

    def prepare_data(self) -> None:
        if len(os.listdir(self.data_dir)) > 0:
            return

        if len(os.listdir(self.raw_data_dir)) == 0:
            self.scrape_data()

        self.extract_data(os.listdir(self.raw_data_dir))
