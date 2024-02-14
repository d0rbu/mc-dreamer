import os
import json


def generate_metadata(
    outputs_dir: str | os.PathLike = "outputs",
    metadata_file: str | os.PathLike = "metadata.json",
    score_threshold: float = 0.7,
) -> None:
    # TODO: walk through outputs_dir and generate metadata.json
    # metadata.json should be a json of the following format
    # {
    #     "total_samples": int,
    #     "score_threshold": float,
    #     "files": [
    #         {
    #             "path": str,
    #         },
    #         ...
    #     ]
    # }

    total_samples = 0
    files_list = []

    for root, _, files in os.walk(outputs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, outputs_dir)
            files_list.append({
                "path": relative_path
            })
            total_samples += 1

    metadata = {
        "total_samples": total_samples,
        "score_threshold": score_threshold,
        "files": files_list
    }

    with open(metadata_file, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    return None
    
    raise NotImplementedError("generate_metadata not implemented yet")
