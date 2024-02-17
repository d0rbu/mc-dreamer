import os
import json
import argparse
import torch as th
from core.data.samples_dataset import DatasetMetadata, FileMetadata


def sum_sizes(files: list[FileMetadata]) -> int:
    return sum(file.size for file in files)

# Function to allocate files to a dataset list until it reaches its target size
def allocate_files(
    target_size: int,
    dataset_list: list[str | os.PathLike],
    remaining_files: list[FileMetadata]
) -> None:
    for i, file in enumerate(remaining_files):
        size_difference = target_size - sum_sizes(dataset_list)
        if size_difference <= 0:
            break

        if file["size"] > size_difference:
            continue

        dataset_list.append(file)
        remaining_files.pop(i)

def generate_metadata(
    outputs_dir: str | os.PathLike = "outputs",
    metadata_file: str | os.PathLike = "metadata.json",
    score_threshold: float = 0.7,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),  # train, val, test
) -> None:
    # metadata.json should be a json of the following format
    # {
    #     "score_threshold": float,
    #     "num_train_samples": int,
    #     "num_val_samples": int,
    #     "num_test_samples": int,
    #     "train": [
    #         {
    #             "path": str,
    #             "size": int,
    #         },
    #         ...
    #     ],
    #     "val": [
    #         {
    #             "path": str,
    #             "size": int,
    #         },
    #         ...
    #     ],
    #     "test": [
    #         {
    #             "path": str,
    #             "size": int,
    #         },
    #         ...
    #     ],
    # }

    assert sum(split_ratios) == 1, "split_ratios must sum to 1"

    # Collect all files with their sizes (in number of samples)
    all_files = []
    for root, _, files in os.walk(outputs_dir):
        if not files:
            continue

        for file in files:
            if not file.endswith(".pt"):
                continue

            file_path = os.path.join(root, file)
            samples = th.load(file_path)
            relative_path = os.path.relpath(file_path, outputs_dir)
            num_samples = (samples["scores"] > score_threshold).sum().item()
            all_files.append(FileMetadata(path=relative_path, size=num_samples))

            del samples

    # Sort files by size in descending order (assuming larger size means more samples)
    all_files.sort(key=lambda x: x.size, reverse=True)

    # Allocate files to train, val, and test sets
    total_samples = len(all_files)
    val_files, test_files = [], []

    # Allocate to test
    target_test_size = int(split_ratios[2] * total_samples)
    allocate_files(target_test_size, test_files, all_files)

    # Allocate to val
    target_val_size = int(split_ratios[1] * total_samples)
    allocate_files(target_val_size, val_files, all_files)

    # Remaining files are allocated to train
    train_files = all_files

    # Prepare metadata
    metadata = DatasetMetadata(
        score_threshold=score_threshold,
        num_train_samples=sum_sizes(train_files),
        num_val_samples=sum_sizes(val_files),
        num_test_samples=sum_sizes(test_files),
        train=train_files,
        val=val_files,
        test=test_files,
    )

    with open(metadata_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--metadata_file", type=str, default="metadata.json")
    args = parser.parse_args()

    generate_metadata(
        outputs_dir=args.outputs_dir,
        metadata_file=args.metadata_file,
        score_threshold=args.threshold
    )
