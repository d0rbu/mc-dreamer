import os
import yaml
import numpy as np
from tqdm import tqdm
from itertools import product
from typing import Sequence, Callable


Heuristic = Callable[[np.ndarray], float]
NUM_BLOCK_TYPES = 256


_disabled_heuristics: set[Heuristic] = set()


def DisableForTesting(heuristic: Heuristic) -> Heuristic:
    _disabled_heuristics.add(heuristic)
    return heuristic

# Getting new weight vectors from YAML files
HEURISTIC_WEIGHTS_DIR = "core/heuristics/heuristic_weights"
INTERESTING_WEIGHTS_PATH = os.path.join(HEURISTIC_WEIGHTS_DIR, "interesting_weights.yaml")
with open(INTERESTING_WEIGHTS_PATH) as f:
    interesting_weights = np.array(yaml.load(f, Loader=yaml.FullLoader))

CRAZY_WEIGHTS_PATHS = os.path.join(HEURISTIC_WEIGHTS_DIR, "too_crazy_weights.yaml")
with open(CRAZY_WEIGHTS_PATHS) as f:
    crazy_weights = np.array(yaml.load(f, Loader=yaml.FullLoader))

class Heuristics:
    @staticmethod
    def num_solid_blocks(sample: np.ndarray) -> float:
        return float(np.mean(sample > 0))

    @staticmethod
    def interesting(sample: np.ndarray) -> float:
        return float(interesting_weights[sample].mean())

    @staticmethod
    def unique_block_ratio(sample: np.ndarray) -> float:
        return len(np.unique(sample)) / NUM_BLOCK_TYPES

    @staticmethod
    def heuristic_1(
        sample: np.ndarray,
        num_solid_weight: float = 1.0,
        interesting_weight: float = 4.0,
        unique_block_ratio_weight: float = 10.0,
    ) -> float:
        return Heuristics.num_solid_blocks(sample) * num_solid_weight + \
               Heuristics.interesting(sample) * interesting_weight + \
               Heuristics.unique_block_ratio(sample) * unique_block_ratio_weight

    @staticmethod
    def interesting_solid_ratio(
        sample: np.ndarray,
    ) -> float:
        interesting_block_score = Heuristics.interesting(sample)

        total_blocks = Heuristics.num_solid_blocks(sample)

        return interesting_block_score / total_blocks if total_blocks > 0 else 0

    @staticmethod
    def interesting_unique_balance(
        sample: np.ndarray,
        interesting_solid_weight: float = 1.0,
        unique_block_weight: float = 1.0,
    ) -> float:
        return min(unique_block_weight * Heuristics.unique_block_ratio(sample), interesting_solid_weight * Heuristics.interesting_solid_ratio(sample))

    @staticmethod
    def fewer_blocks(
        sample: np.ndarray,
        good_ratio: float = 14/256,
        bad_ratio: float = 50/256,
    ) -> float:
        unique_block_ratio = Heuristics.unique_block_ratio(sample)
        if unique_block_ratio <= good_ratio:
            return unique_block_ratio / good_ratio
        elif unique_block_ratio >= bad_ratio:
            return 1 - (unique_block_ratio - bad_ratio)

        return 1

    @DisableForTesting
    @staticmethod
    def mix(
        sample: np.ndarray,
        heuristic_weights: dict[str, float]
    ) -> float:
        # allows for dynamic mixing of heuristics
        heuristics = {
            name: (getattr(Heuristics, name), heuristic_weight) for name, heuristic_weight in heuristic_weights.items()
        }

        score = 0
        for name, (heuristic, heuristic_weight) in heuristics.items():
            score += heuristic_weight * heuristic(sample)

        return score


class OptimizedHeuristics:
    @staticmethod
    def num_solid_blocks(sample: np.ndarray) -> float:
        # sample is now a 256-dimensional vector containing block counts
        return float(sample[1].sum()) / 4096

    @staticmethod
    def interesting(sample: np.ndarray) -> float:
        return float(interesting_weights.dot(sample)) / 4096

    @staticmethod
    def unique_block_ratio(sample: np.ndarray) -> float:
        return (sample > 0).sum() / NUM_BLOCK_TYPES

    @DisableForTesting
    @staticmethod
    def interesting_solid_ratio(sample: np.ndarray) -> float:
        interesting_block_score = float(interesting_weights[sample].mean())
        total_blocks = (sample > 0).sum()

        return interesting_block_score / total_blocks if total_blocks > 0 else 0

    @staticmethod
    def interesting_unique_balance(
        sample: np.ndarray,
        interesting_solid_weight: float = 1.0,
        unique_block_weight: float = 1.0,
    ) -> float:
        return min(unique_block_weight * OptimizedHeuristics.unique_block_ratio(sample), interesting_solid_weight * OptimizedHeuristics.interesting_solid_ratio(sample))

    @staticmethod
    def fewer_blocks(
        sample: np.ndarray,
        good_ratio: float = 14/256,
        bad_ratio: float = 50/256,
    ) -> float:
        unique_block_ratio = OptimizedHeuristics.unique_block_ratio(sample)
        if unique_block_ratio <= good_ratio:
            return unique_block_ratio / good_ratio
        elif unique_block_ratio >= bad_ratio:
            return 1 - (unique_block_ratio - bad_ratio)

        return 1

    @staticmethod
    def best_heuristic(
        sample: np.ndarray,
        sample_size: tuple[int, int, int] = (16, 16, 16),
    ) -> float:
        import pdb; pdb.set_trace()
        # 16^3
        total_blocks = sample_size[0] * sample_size[1] * sample_size[2]

        # Apply weights to chunk
        interesting_score = interesting_weights.dot(sample) / total_blocks

        solid_blocks_ratio = sample[1:].sum() / total_blocks
        interesting_solid_ratio = interesting_score / solid_blocks_ratio if solid_blocks_ratio > 0 else 0

        # def unique_block_ratio(sample: np.ndarray)
        unique_block_ratio = (sample > 0).sum() / NUM_BLOCK_TYPES

        fewer_blocks_score = min(1, unique_block_ratio / (14/256), 1 - (unique_block_ratio - 50/256))

        return \
            interesting_score * 1/6 + \
            interesting_solid_ratio * 1/3 + \
            fewer_blocks_score * 1/2


HEURISTICS: Sequence[Heuristic] = [
    heuristic
    for heuristic in Heuristics.__dict__.values()
    if isinstance(heuristic, Callable) and heuristic not in _disabled_heuristics
]
