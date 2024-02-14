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
            if not file.endswith(".scores"):
                continue
            
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                scores = pickle.load(f)
            
            for i, score in enumerate(scores):
                full_scores.append((file_path, i, score))

    full_scores = sorted(full_scores.items(), key=lambda x: x[1])
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

        for file in files:
            if not file.endswith(".scores"):
                continue
            
            file_path = os.path.join(root, file)
            import pdb; pdb.set_trace()
            with open(file_path, "rb") as f:
                scores = pickle.load(f)
            
            num_samples = len(scores)
            selected_indices = np.random.choice(num_samples, min(samples_per_file, num_samples), replace=False)
            
            for i in range(selected_indices):
                full_scores.append((file_path, i, scores[i]))
    
    full_scores = sorted(full_scores.items(), key=lambda x: x[1])
    num_samples = len(full_scores)
    percentiles = percentiles * num_samples
    percentiles[-1] = num_samples - 1
    percentiles = percentiles.to(int)
    percentiles = percentiles.unique()

    selected_scores = [full_scores[i] for i in percentiles]

    return selected_scores

def convert_percentiles(
    percentiles: th.Tensor,
    name: str | None = None,
    extraction_output_dir: str | os.PathLike = "outputs",
    schematic_dir: str | os.PathLike = "schematics_output",
    randomly_sample: bool = True,
    num_external_sort_files: int = 0,  # max number of files at a time for external sort
) -> None:
    if name is not None:
        extraction_output_dir = os.path.join(extraction_output_dir, name)

    if randomly_sample:
        selected_scores = get_percentile_scores_random(extraction_output_dir, percentiles, num_external_sort_files)
    elif num_external_sort_files > 0:
        selected_scores = get_percentile_scores_external(extraction_output_dir, percentiles, num_external_sort_files)
    else:
        selected_scores = get_percentile_scores(extraction_output_dir, percentiles)

    loaded_samples = {}

    for (scores_path, i), score in tqdm(selected_scores, desc="Gathering schematics"):
        file_path = scores_path.replace(".scores", ".pt")

        if file_path in loaded_samples:
            samples = loaded_samples[file_path]
        else:
            samples = th.load(file_path)
            loaded_samples[file_path] = samples
        
        sample = samples[i]
        sf = SchematicFile(shape=sample.shape)
        sf.blocks = sample.numpy().astype(np.uint8).transpose(1, 2, 0)

        sf.save(os.path.join(schematic_dir, f"{score:.5f}.schematic"))

    print("Done!")

if __name__ == "__main__":
    percentiles = th.linspace(0, 1, 101)

    convert_percentiles(percentiles)
