import os
import pickle
import numpy as np
import torch as th
from tempfile import TemporaryDirectory
from nbtschematic import SchematicFile
from tqdm import tqdm

def get_percentile_scores(
    extraction_output_dir: str | os.PathLike,
    percentiles: th.Tensor,
) -> list[tuple[str, int, float]]:
    full_scores: list[tuple[str, int, float]] = []  # (file, index, score)

    for root, subdirs, files in tqdm(os.walk(extraction_output_dir), desc="Gathering scores"):
        if not files:
            continue

        for file in files:
            if not file.endswith(".pt"):
                continue

            file_path = os.path.join(root, file)
            raw_data = th.load(file_path, map_location=th.device("cpu"))
            scores = raw_data["scores"].flatten()

            for i, score in enumerate(scores):
                full_scores.append((file_path, i, score))

    full_scores = sorted(full_scores, key=lambda x: x[1])
    num_samples = len(full_scores)
    percentiles = percentiles * num_samples
    percentiles[-1] = num_samples - 1
    percentiles = percentiles.to(int)
    percentiles = percentiles.unique()

    selected_scores = [full_scores[i] for i in percentiles]
    
    return selected_scores

def get_percentile_scores_external(
    extraction_output_dir: str | os.PathLike,
    percentiles: th.Tensor,
    num_external_sort_files: int,
) -> list[tuple[str, int, float]]:
    raise NotImplementedError("External sort not implemented yet")

def get_percentile_scores_random(
    extraction_output_dir: str | os.PathLike,
    percentiles: th.Tensor,
    samples_per_file: int = 10,
) -> list[tuple[str, int, float]]:
    full_scores: list[tuple[str, int, float]] = []  # (file, index, score)

    for root, subdirs, files in tqdm(os.walk(extraction_output_dir), desc="Randomly gathering scores"):
        if not files:
            continue

        for file in tqdm(files, total=len(files), leave=False):
            if not file.endswith(".pt"):
                continue

            file_path = os.path.join(root, file)
            raw_data = th.load(file_path, map_location=th.device("cpu"))
            scores = raw_data["scores"].flatten()

            random_indices = np.random.choice(len(scores), min(samples_per_file, len(scores)), replace=False)

            for i in random_indices:
                full_scores.append((file_path, i, scores[i]))

    full_scores = sorted(full_scores.items(), key=lambda x: x[1])
    num_samples = len(full_scores)
    percentiles = percentiles * num_samples
    percentiles[-1] = num_samples - 1
    percentiles = percentiles.to(int)
    percentiles = percentiles.unique()

    selected_scores = [full_scores[i] for i in percentiles]

    return selected_scores

VOLUME_SIZE = (16, 16, 16)

def convert_percentiles(
    percentiles: th.Tensor,
    name: str | None = None,
    extraction_output_dir: str | os.PathLike = "outputs",
    schematic_dir: str | os.PathLike = "schematics_output",
    num_sampled: int = 0,  # number of files to sample
    num_external_sort_files: int = 0,  # max number of files at a time for external sort
) -> None:
    if name is not None:
        extraction_output_dir = os.path.join(extraction_output_dir, name)

    if num_sampled > 0:
        selected_scores = get_percentile_scores_random(extraction_output_dir, percentiles, num_sampled)
    elif num_external_sort_files > 0:
        selected_scores = get_percentile_scores_external(extraction_output_dir, percentiles, num_external_sort_files)
    else:
        selected_scores = get_percentile_scores(extraction_output_dir, percentiles)

    loaded_samples = {}

    for file_path, i, score in tqdm(selected_scores, desc="Gathering schematics"):
        if file_path in loaded_samples:
            samples = loaded_samples[file_path]
        else:
            samples = th.load(file_path)["region"]
            loaded_samples[file_path] = samples

        x, y, z = np.unravel_index(i, samples.shape)
        sample = samples[x:x+VOLUME_SIZE[0], y:y+VOLUME_SIZE[1], z:z+VOLUME_SIZE[2]]
        sf = SchematicFile(shape=sample.shape)
        sf.blocks = sample.numpy().astype(np.uint8).transpose(1, 2, 0)

        sf.save(os.path.join(schematic_dir, f"{score:.5f}.schematic"))

    print("Done!")

if __name__ == "__main__":
    percentiles = th.linspace(0, 1, 101)

    convert_percentiles(percentiles, num_sampled=8)
