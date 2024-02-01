import os
import yaml
import math


def ref_scores_path() -> str:
    return "ref_scores.yaml"

def get_ref_scores() -> dict[str, dict[str, float]]:
    path = ref_scores_path()
    if not os.path.exists(path):
        return {}

    with open(ref_scores_path()) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def set_ref_score(
    name: str,
    scores: dict[str, float],
) -> None:
    ref_scores = get_ref_scores()
    ref_scores[name] = scores

    with open(ref_scores_path(), "w") as f:
        yaml.dump(ref_scores, f)

def normalize_score(
    scores: dict[str, float],
) -> dict[str, float]:
    min_score = min(scores.values())
    max_score = max(scores.values())

    if min_score == max_score:
        return {name: 0.5 for name in scores.keys()}

    return {
        name: (score - min_score) / (max_score - min_score)
        for name, score in scores.items()
    }

def normalize_scores(
    scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        name: normalize_score(score)
        for name, score in scores.items()
    }

def average_ref_scores(
    ref_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    scored_files = set.union(*[set(scores.keys()) for scores in ref_scores.values()])

    average_scores = {
        file: 0.0
        for file in scored_files
    }
    for file in scored_files:
        num_scores = 0
        for scores in ref_scores.values():
            if file not in scores:
                continue

            average_scores[file] += scores[file]
            num_scores += 1

        average_scores[file] /= num_scores

    return average_scores

def score_binary_cross_entropy(
    base_scores: dict[str, float],
    approximated_scores: dict[str, float],
    eps: float = 1e-6,
) -> float | None:
    shared_files = set(base_scores.keys()) & set(approximated_scores.keys())

    if len(shared_files) == 0:
        return None

    return -sum(
        base_scores[file] * math.log(approximated_scores[file] + eps) + (1 - base_scores[file]) * math.log(1 - approximated_scores[file] + eps)
        for file in shared_files
    ) / len(shared_files)
