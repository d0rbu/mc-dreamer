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

    HEURISTIC_WEIGHT_VALUES = (
        ("interesting", np.linspace(0.1, 0.22, 10)),
        ("interesting_solid_ratio", np.linspace(0.26, 0.39, 10)),
        ("fewer_blocks", np.linspace(0.45, 0.55, 10)),
    )

    heuristic_names = [name for name, _ in HEURISTIC_WEIGHT_VALUES]
    heuristic_steps = [steps for _, steps in HEURISTIC_WEIGHT_VALUES]

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
    for name, steps in HEURISTIC_WEIGHT_VALUES:
        total_steps *= int(steps.shape[0])

    heuristic_weight_candidates = product(*heuristic_steps)
    consume(heuristic_weight_candidates, 1)
    for heuristic_weights in tqdm(heuristic_weight_candidates, total=total_steps):
        weight_sum = sum(heuristic_weights)
        weights = {
            name: weight / weight_sum for name, weight in zip(heuristic_names, heuristic_weights)
        }
        scores = {
            file: Heuristics.mix(sample, weights)
            for file, sample in samples.items()
        }
        normalized_scores = normalize_score(scores)

        loss = score_binary_cross_entropy(avg_ref_scores, normalized_scores)

        if min_loss > loss:
            min_loss = loss
            min_weights = weights
            print(f"new min: {min_loss}\nweights: {min_weights}")

    print(min_weights)

    return min_loss


if __name__ == "__main__":
    test_heuristics()
