import os
import pickle
import numpy as np
import torch as th
from nbtschematic import SchematicFile
from tqdm import tqdm

def convert_percentiles(
    percentiles: th.Tensor,
    name: str | None = None,
    extraction_output_dir: str | os.PathLike = "outputs",
    schematic_dir: str | os.PathLike = "schematics_output",
) -> None:
    full_scores: dict[tuple[str, int], float] = {}  # maps (file, index) to score

    if name is not None:
        extraction_output_dir = os.path.join(extraction_output_dir, name)

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
                full_scores[(file_path, i)] = score
    
    sorted_scores = sorted(full_scores.items(), key=lambda x: x[1])
    num_samples = len(sorted_scores)
    percentiles = percentiles * num_samples
    percentiles[-1] = num_samples - 1
    percentiles = percentiles.to(int)
    percentiles = percentiles.unique()

    loaded_samples = {}

    for percentile in tqdm(percentiles, desc="Gathering schematics"):
        (scores_path, i), score = sorted_scores[percentile]
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
