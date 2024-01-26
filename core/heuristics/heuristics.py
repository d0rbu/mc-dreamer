import numpy as np
from typing import Sequence, Callable


Heuristic = Callable[[np.ndarray], float]
NUM_BLOCKS = 256


_disabled_heuristics: set[Heuristic] = set()


def DisableForTesting(heuristic: Heuristic) -> Heuristic:
    _disabled_heuristics.add(heuristic)
    return heuristic


class Heuristics:
    @staticmethod
    def num_blocks(sample: np.ndarray) -> float:
        return np.mean(sample > 0)

    @staticmethod
    def block_weighted(sample: np.ndarray) -> float:
        block_weights = np.ones(NUM_BLOCKS)
        block_weights[0] = 0  # air is uninteresting
        block_weights[1:5] = 0.1  #  stone, grass, dirt, cobble
        block_weights[6:8] = 0.1  # sapling, bedrock
        block_weights[9] = 0.1  # still water
        block_weights[11] = 0.2  # still lava
        block_weights[12:17] = 0.1  # sand, gravel, and ores
        block_weights[17] = 0.3  # wood
        block_weights[18] = 0.1  # leaves
        block_weights[21] = 0.1  # lapis ore
        block_weights[24] = 0.1  # sandstone
        block_weights[30:33] = 0.1  # cobweb, grass
        block_weights[37:41] = 0.2  # flower, mushroom
        block_weights[48] = 0.3  # mossy cobble
        block_weights[49] = 0.1  # obsidian
        block_weights[52] = 0.3  # spawner
        block_weights[56] = 0.3  # diamond ore
        block_weights[73] = 0.1  # redstone ore
        block_weights[78:83] = 0.1  # snow, ice, snow block, cactus, clay
        block_weights[83] = 0.2  # sugar cane
        block_weights[86:89] = 0.1  # pumpkin, netherrack, soul sand
        block_weights[89] = 0.3  # glowstone
        block_weights[97] = 0.1  # monster egg
        block_weights[98] = 0.2  # stone brick
        block_weights[99:101] = 0.2  # brown mushroom, red mushroom
        block_weights[101:103] = 0.5  # iron bars, glass pane
        block_weights[103:106] = 0.1  # melon, pumpkin stem, melon stem
        block_weights[106] = 0.2  # vine
        block_weights[110] = 0.1  # mycelium
        block_weights[111] = 0.3  # lily pad
        block_weights[121] = 0.1  # end stone
        block_weights[127:129] = 0.1  # cocoa, sandstone stairs
        block_weights[129] = 0.2  # emerald ore
        block_weights[139] = 0.5  # cobblestone wall
        block_weights[141:143] = 0.3  # carrot, potato
        block_weights[145] = 0.6  # anvil
        block_weights[146] = 0.5  # trapped chest
        block_weights[153] = 0.1  # nether quartz ore
        block_weights[159] = 0.2  # stained clay
        block_weights[161] = 0.1  # acacia leaves
        block_weights[162] = 0.3  # acacia wood
        block_weights[168] = 0.2  # prismarine
        block_weights[170] = 0.2  # hay bale
        block_weights[171] = 0.5  # carpet
        block_weights[172] = 0.3  # hardened clay
        block_weights[174] = 0.2  # packed ice
        block_weights[175] = 0.2  # double plant
        block_weights[179:183] = 0.1  # red sandstone
        block_weights[212:214] = 0.2  # frosted ice, magma block

        return block_weights[sample].mean()
    
    @staticmethod
    def num_unique_blocks(sample: np.ndarray) -> float:
        return len(np.unique(sample)) / NUM_BLOCKS
    
    @staticmethod
    def heuristic_1(
        sample: np.ndarray,
        num_blocks_weight: float = 1.0,
        block_weighted_weight: float = 4.0,
        num_unique_blocks_weight: float = 10.0,
    ) -> float:
        return Heuristics.num_blocks(sample) * num_blocks_weight + \
               Heuristics.block_weighted(sample) * block_weighted_weight + \
               Heuristics.num_unique_blocks(sample) * num_unique_blocks_weight


HEURISTICS: Sequence[Heuristic] = [
    heuristic
    for heuristic in Heuristics.__dict__.values()
    if isinstance(heuristic, Callable) and heuristic not in _disabled_heuristics
]
