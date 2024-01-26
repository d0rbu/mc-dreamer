import os
import yaml
import numpy as np
from core.heuristics.heuristics import HEURISTICS
from core.heuristics.ref_scores import (
    normalize_ref_scores,
    average_ref_scores,
    get_ref_scores,
    score_binary_cross_entropy,
)
from typing import Sequence


def _pprint_scores(
    scores: dict[str, dict[str, float]],
    files: list[str],
) -> None:
    longest_file_name = max(len(file) for file in files)
    longest_heuristic_name = max(len(heuristic) for heuristic in scores.keys())

    print(" " * (longest_file_name + 3), end="")
    for heuristic in HEURISTICS:
        print(f"{heuristic.__name__}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")

    print("\b\b ")

    for file in files:
        print(f"{file}{' ' * (longest_file_name - len(file))} | ", end="")
        for heuristic in HEURISTICS:
            print(f"{scores[heuristic.__name__][file]:.4f}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")

        print("\b\b ")

def _pprint_comparisons(
    ref_score_names: Sequence[str],
    comparisons: dict[str, dict[str, float]],
) -> None:
    longest_heuristic_name = max(len(heuristic.__name__) for heuristic in HEURISTICS)
    longest_ref_score_name = max(len(name) for name in ref_score_names)

    print(" " * (longest_ref_score_name + 3), end="")
    for heuristic in HEURISTICS:
        print(f"{heuristic.__name__}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")

    print("\b\b ")

    for ref_score_name in ref_score_names:
        print(f"{ref_score_name}{' ' * (longest_ref_score_name - len(ref_score_name))} | ", end="")
        for heuristic in HEURISTICS:
            binary_cross_entropy = comparisons[heuristic.__name__][ref_score_name]
            if binary_cross_entropy is None:
                score_string = "NaN "
            else:
                score_string = f"{binary_cross_entropy:.4f}"

            print(f"{score_string}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")

        print("\b\b ")

AVG_REF_SCORE_NAME = "avg"

def test_heuristics(
    output_dir: str | os.PathLike = "test_outputs",
    input_dir: str | os.PathLike = "test_inputs",
) -> None:
    files = [file for file in os.listdir(input_dir) if file.endswith(".npy")]

    assert len(files) > 0, f"no .npy files found in {input_dir}"

    scores = {
        heuristic.__name__: {
            file: 0.0
            for file in files
        }
        for heuristic in HEURISTICS
    }

    for file in files:
        file_path = os.path.join(input_dir, file)
        np_sample = np.load(file_path)

        for heuristic in HEURISTICS:
            scores[heuristic.__name__][file] = heuristic(np_sample)

    yaml.dump(scores, open(os.path.join(output_dir, "heuristics_scores.yaml"), "w"))
    
    _pprint_scores(scores, files)

    ref_scores = get_ref_scores()
    normalized_ref_scores = normalize_ref_scores(get_ref_scores())
    avg_ref_scores = average_ref_scores(normalized_ref_scores)
    if AVG_REF_SCORE_NAME in ref_scores:
        print(f"Warning: somebody's reference scores are already named 'avg', overwriting them")
        del ref_scores[AVG_REF_SCORE_NAME]

    ref_names = list(ref_scores.keys())
    ref_names.insert(0, AVG_REF_SCORE_NAME)

    ref_scores[AVG_REF_SCORE_NAME] = avg_ref_scores

    comparisons = {
        heuristic.__name__: {
            ref_name: score_binary_cross_entropy(ref_scores[ref_name], scores[heuristic.__name__])
            for ref_name in ref_names
        }
        for heuristic in HEURISTICS
    }

    _pprint_comparisons(ref_names, comparisons)

    yaml.dump(comparisons, open(os.path.join(output_dir, "heuristics_comparisons.yaml"), "w"))


if __name__ == "__main__":
    test_heuristics()
