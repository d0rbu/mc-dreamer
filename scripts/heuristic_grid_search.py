import os
import yaml
import numpy as np
from itertools import product
from more_itertools import consume
from tqdm import tqdm
from core.heuristics.heuristics import HEURISTICS, Heuristics
from core.heuristics.ref_scores import (
    normalize_scores,
    normalize_score,
    average_ref_scores,
    get_ref_scores,
    score_binary_cross_entropy,
)
from typing import Sequence

SCORE_WIDTH = 6
AVG_REF_SCORE_NAME = "avg"

def test_heuristics(
    output_dir: str | os.PathLike = "test_outputs",
    input_dir: str | os.PathLike = "test_inputs",
) -> None:
    files = [file for file in os.listdir(input_dir) if file.endswith(".npy")]

    assert len(files) > 0, f"no .npy files found in {input_dir}"

    ref_scores = get_ref_scores()
    normalized_ref_scores = normalize_scores(ref_scores)
    avg_ref_scores = average_ref_scores(normalized_ref_scores)

    HEURISTIC_WEIGHT_STEPS = (
        ("num_blocks", 10),
        ("block_weighted", 10),
        ("num_unique_blocks", 10),
        ("intresing_decency", 10),
        ("fewer_blocks", 10),
    )

    heuristic_names = [name for name, _ in HEURISTIC_WEIGHT_STEPS]
    heuristic_steps = [np.linspace(0, 1, steps) for _, steps in HEURISTIC_WEIGHT_STEPS]

    samples = {}
    for file in files:
        file_path = os.path.join(input_dir, file)
        np_sample = np.load(file_path)

        samples[file] = np_sample

    # Make Dictionary and do grid search
    min_loss = float("inf")
    min_weights = {
        heuristic_name: None for heuristic_name in heuristic_names
    }
    total_steps = 1
    for name, steps in HEURISTIC_WEIGHT_STEPS:
        total_steps *= steps

    heuristic_weight_candidates = product(*heuristic_steps)
    consume(heuristic_weight_candidates, 1)
    for heuristic_weights in tqdm(heuristic_weight_candidates, total=total_steps):
        weights = {
            name: weight for name, weight in zip(heuristic_names, heuristic_weights)
        }
        scores = {
            file: Heuristics.mix(sample, weights)
            for file, sample in samples.items()
        }
        normalized_scores = normalize_score(scores)

        loss = score_binary_cross_entropy(avg_ref_scores, normalized_scores)
        
        if min_loss > loss:
            min_loss = loss
            min_weights = {
                heuristic_name: heuristic_weight for heuristic_name, heuristic_weight in zip(heuristic_names, heuristic_weights)
            }
            print(f"new min: {min_loss}\nweights: {min_weights}")

    print(min_weights)

    return min_loss


if __name__ == "__main__":
    test_heuristics()
