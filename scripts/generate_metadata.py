import os
import json
import argparse


def generate_metadata(
    outputs_dir: str | os.PathLike = "outputs",
    metadata_file: str | os.PathLike = "metadata.json",
    score_threshold: float = 0.7,
) -> None:
    # metadata.json should be a json of the following format
    # {
    #     "score_threshold": float,
    #     "files": [
    #         {
    #             "path": str,
    #         },
    #         ...
    #     ]
    # }

    files_list = []

    for root, _, files in os.walk(outputs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, outputs_dir)
            files_list.append({
                "path": relative_path
            })

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
