import os
import yaml
import numpy as np
from itertools import product
from typing import Sequence, Callable

Filter = Callable[[np.ndarray], float]
NUM_BLOCK_TYPES = 256

class TestFilters:
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

    @staticmethod
    def mix(
        sample: np.ndarray,
        heuristic_weights: dict[str, float] = {
            "interesting": 1/6,
            "interesting_solid_ratio": 1/3,
            "fewer_blocks": 1/2,
        }
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
    interesting_weights = interesting_weights

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
        interesting_block_score = OptimizedHeuristics.interesting(sample)
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
        sample: np.ndarray,  # (*, 256) block count vector(s)
        sample_size: tuple[int, int, int] = (16, 16, 16),
    ) -> np.ndarray:
        # 16^3
        total_blocks = sample_size[0] * sample_size[1] * sample_size[2]

        original_shape = sample.shape
        sample = sample.reshape(-1, 256)
        interesting_score = sample @ OptimizedHeuristics.interesting_weights / total_blocks

        solid_blocks_ratio = sample[:, 1:].sum(axis=-1) / total_blocks

        air_mask = solid_blocks_ratio == 0
        interesting_solid_ratio = np.empty_like(solid_blocks_ratio)
        interesting_solid_ratio[air_mask] = 0
        interesting_solid_ratio[~air_mask] = interesting_score[~air_mask] / solid_blocks_ratio[~air_mask]

        unique_block_ratio = (sample > 0).sum(axis=-1) / NUM_BLOCK_TYPES

        lower_bound_unique_block_ratios = unique_block_ratio / (14/256)
        upper_bound_unique_block_ratios = 1 - (unique_block_ratio - 50/256)
        ones = np.ones_like(unique_block_ratio)
        fewer_blocks_options = np.stack((ones, lower_bound_unique_block_ratios, upper_bound_unique_block_ratios), axis=-1)
        fewer_blocks_score = fewer_blocks_options.min(axis=-1)

        score = interesting_score * 1/6 + \
                interesting_solid_ratio * 1/3 + \
                fewer_blocks_score * 1/2
        
        return score.reshape(original_shape[:-1])


FILTERS: Sequence[Filter] = [
    heuristic
    for heuristic in Filter.__dict__.values()
    if isinstance(heuristic, Callable) and heuristic not in _disabled_heuristics
]  