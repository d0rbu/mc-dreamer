import os
import json
import argparse
from core.data.samples_dataset import DatasetMetadata, FileMetadata


def generate_metadata(
    outputs_dir: str | os.PathLike = "outputs",
    metadata_file: str | os.PathLike = "metadata.json",
    score_threshold: float = 0.7,
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
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
    #         },
    #         ...
    #     ],
    #     "val": [
    #         {
    #             "path": str,
    #         },
    #         ...
    #     ],
    #     "test": [
    #         {
    #             "path": str,
    #         },
    #         ...
    #     ],
    # }

    assert sum(split_ratios) == 1, "split_ratios must sum to 1"

    # Collect all files with their sizes (as a proxy for number of samples)
    all_files = []
    for root, _, files in os.walk(outputs_dir):
        for file in files:
            passed_scores =
            while
            file_size = file_path.stat().st_size  # Using file size as a proxy
            all_files.append({"path": str(file_path.relative_to(outputs_dir)), "size": file_size})

    # Sort files by size in descending order (assuming larger size means more samples)
    all_files.sort(key=lambda x: x['size'], reverse=True)

    # Allocate files to train, val, and test sets
    total_samples = len(all_files)
    train_list, val_list, test_list = [], [], []
    allocated_samples = 0

    # Function to allocate files to a dataset list until it reaches its target size
    def allocate_files(target_size, dataset_list, remaining_files):
        while remaining_files and len(dataset_list) < target_size:
            dataset_list.append(remaining_files.pop(0))

    # Allocate to train
    target_train_size = int(split_ratios[0] * total_samples)
    allocate_files(target_train_size, train_list, all_files)

    # Allocate to test
    target_test_size = int(split_ratios[1] * total_samples)
    allocate_files(target_test_size, test_list, all_files)

    # Remaining files go to val
    val_list.extend(all_files)  # Whatever remains goes to val

    # Prepare metadata
    metadata = {
        "score_threshold": score_threshold,
        "num_train_samples": len(train_list),
        "num_val_samples": len(val_list),
        "num_test_samples": len(test_list),
        "train": [{"path": file['path']} for file in train_list],
        "val": [{"path": file['path']} for file in val_list],
        "test": [{"path": file['path']} for file in test_list],
    }

    with open(metadata_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


    # sort files by # of samples
    # while len(train_list) < split_ratios[0] * total_samples:
    #     # search for largest file we can fit into train without going over
    #     # pop it and add it to train
    #     # repeat
    # 
    # repeat for test and then val
    # make sure we do this in (train, test, val) order because we want to prioritize train, then test, then val

    
    sorted_files = []
    for root, _, files in os.walk(outputs_dir):
        for file in files:
            sorted_files.append(file)
            '''file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, outputs_dir)
            files_list.append({
                "path": relative_path


            })'''
    sorted_files.sort()
    while len(train_list) < split_ratios[0] * total_samples:

    metadata = {
        "score_threshold": score_threshold,
        "files": files_list
    }

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
