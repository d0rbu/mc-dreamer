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

    train_list = []
    val_list = []
    test_list = []

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
