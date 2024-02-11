import os
import zipfile
import json
import anvil.anvil as anvil
import numpy as np
import torch as th
from tqdm import tqdm
from uuid import uuid4
from mpi4py import MPI
from itertools import chain, product
from more_itertools import chunked
from typing import Sequence
from core.heuristics.heuristics import OptimizedHeuristics


CHUNK_LENGTH = 16  # in blocks
CHUNK_HEIGHT = 256  # in blocks
REGION_LENGTH = 32  # in chunks
REGION_BLOCK_LENGTH = REGION_LENGTH * CHUNK_LENGTH  # in blocks
NUM_BLOCKS = 256
BLOCK_MAP_PATH = "core/extract/block_map.json"

with open(BLOCK_MAP_PATH, "r") as f:
    BLOCK_MAP = json.load(f)
    BLOCK_MAP = {
        f"minecraft:{name}": id for name, id in BLOCK_MAP.items()
    }
    underscore_block_map = {
        name.replace(" ", "_"): id for name, id in BLOCK_MAP.items()
    }
    BLOCK_MAP.update(underscore_block_map)

COMM = MPI.COMM_WORLD

def block_name_to_id(block_name: str) -> int:
    return anvil.Block.from_name(block_name).id()

def get_available_regions(region_dir: str | os.PathLike) -> tuple[np.ndarray, Sequence[tuple[int, int]]]:
    region_files = [file for file in os.listdir(region_dir) if file.endswith(".mca")]
    available_region_x = []
    available_region_z = []
    for region_file in region_files:
        region_x, region_z = region_file.split(".")[1:3]
        available_region_x.append(int(region_x))
        available_region_z.append(int(region_z))

    min_x = min(available_region_x)
    max_x = max(available_region_x)
    min_z = min(available_region_z)
    max_z = max(available_region_z)

    available_regions = np.zeros((abs(min_x) + abs(max_x), abs(min_z) + abs(max_z)), dtype=bool)
    available_regions[available_region_x, available_region_z] = True

    return available_regions, zip(available_region_x, available_region_z)

def extract_from_ndarray(
    dense_array: np.ndarray,
    sample_size: tuple[int, int, int],
    score_threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[float]]:
    extracted_samples = []
    scores = []

    scores = np.empty((dense_array.shape[0] - sample_size[0] + 1, dense_array.shape[1] - sample_size[1] + 1, dense_array.shape[2] - sample_size[2] + 1), dtype=np.float32)
    running_integral = np.zeros((sample_size[0] + 1, *dense_array.shape[1:], NUM_BLOCKS), dtype=np.uintc)

    x_generator = range(dense_array.shape[0])
    if COMM.Get_rank() == 0:
        x_generator = tqdm(x_generator, total=dense_array.shape[0], desc="Scoring samples")

    for x in x_generator:
        running_integral = np.roll(running_integral, shift=-1, axis=0)
        sparse_plane = np.zeros((*dense_array.shape[1:], NUM_BLOCKS), dtype=np.uintc)
        np.put_along_axis(sparse_plane, np.expand_dims(dense_array[x], axis=-1), 1, axis=-1)

        running_integral[-1] = np.cumsum(sparse_plane, axis=0)  # integral
        running_integral[-1] = np.cumsum(running_integral[-1], axis=1)  # double integral
        running_integral[-1] += running_integral[-2]  # triple integral

        if x < sample_size[0] - 1:
            continue

        plane_block_counts = np.empty((dense_array.shape[1] - sample_size[1] + 1, dense_array.shape[2] - sample_size[2] + 1, NUM_BLOCKS), dtype=np.uintc)
        plane_block_counts[1:, 1:] = \
            running_integral[-1, sample_size[1]:, sample_size[2]:] - \
            running_integral[-1, :-sample_size[1], sample_size[2]:] - \
            running_integral[-1, sample_size[1]:, :-sample_size[2]] - \
            running_integral[0, sample_size[1]:, sample_size[2]:] + \
            running_integral[0, :-sample_size[1], sample_size[2]:] + \
            running_integral[0, sample_size[1]:, :-sample_size[2]] + \
            running_integral[-1, :-sample_size[1], :-sample_size[2]] - \
            running_integral[0, :-sample_size[1], :-sample_size[2]]  # dynamic programming baby
        plane_block_counts[0, 1:] = \
            running_integral[-1, sample_size[1] - 1, sample_size[2]:] - \
            running_integral[-1, sample_size[1] - 1, :-sample_size[2]] - \
            running_integral[0, sample_size[1] - 1, sample_size[2]:] + \
            running_integral[0, sample_size[1] - 1, :-sample_size[2]]
        plane_block_counts[1:, 0] = \
            running_integral[-1, sample_size[1]:, sample_size[2] - 1] - \
            running_integral[-1, :-sample_size[1], sample_size[2] - 1] - \
            running_integral[0, sample_size[1]:, sample_size[2] - 1] + \
            running_integral[0, :-sample_size[1], sample_size[2] - 1]
        plane_block_counts[0, 0] = \
            running_integral[-1, sample_size[1] - 1, sample_size[2] - 1] - \
            running_integral[0, sample_size[1] - 1, sample_size[2] - 1]
        
        assert (plane_block_counts.sum(axis=-1) == sample_size[0] * sample_size[1] * sample_size[2]).all()

        scores[x - sample_size[0] + 1] = OptimizedHeuristics.best_heuristic(plane_block_counts, sample_size)

    # now pick scores above a certain threshold and extract the samples
    for x, y, z in tqdm(zip(*np.where(scores > score_threshold)), total=scores.size, leave=False, desc="Extracting samples"):
        extracted_samples.append(dense_array[x:x + sample_size[0], y:y + sample_size[1], z:z + sample_size[2]])
        scores.append(scores[x, y, z])

    return extracted_samples, scores

def extract_from_region(
    file_path: str | os.PathLike,
    sample_size: tuple[int, int, int],
    score_threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[float]]:
    ndarray_filepath = file_path.replace(".mca", ".npy")
    if os.path.exists(ndarray_filepath):
        ndarray_blocks = np.load(ndarray_filepath)
    else:
        with open(file_path, "rb") as region_file:
            region = anvil.Region.from_file(region_file)

        ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=np.ubyte)
        for chunk_x, chunk_z in tqdm(product(range(REGION_LENGTH), repeat=2), total=REGION_LENGTH ** 2, leave=False, desc="Converting region to ndarray"):
            chunk = anvil.Chunk.from_region(region, chunk_x, chunk_z)
            for block_x, block_y, block_z in product(range(CHUNK_LENGTH), range(CHUNK_HEIGHT), range(CHUNK_LENGTH)):
                x, y, z = chunk_x * CHUNK_LENGTH + block_x, block_y, chunk_z * CHUNK_LENGTH + block_z
                block_name = chunk.get_block(block_x, block_y, block_z).name()
                ndarray_blocks[x, y, z] = BLOCK_MAP[block_name]

        np.save(ndarray_filepath, ndarray_blocks)

    return extract_from_ndarray(ndarray_blocks, sample_size, score_threshold)

def get_edge_samples(
    file_paths: tuple[str | os.PathLike, str | os.PathLike],
    axis: int = 0,  # which axis to stitch the samples along, should be 0 or 2
) -> np.ndarray:
    regions = []
    for file_path in file_paths:
        with open(file_path, "rb") as region_file:
            regions.append(anvil.Region.from_file(region_file))

    region_extracted_size = [REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH]
    region_extracted_size[axis] = CHUNK_LENGTH
    
    # first_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    # second_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    first_ndarray_region = np.empty(*region_extracted_size, dtype=object)
    second_ndarray_region = np.empty(*region_extracted_size, dtype=object)

    for block_x, block_y, block_z in product(*[range(-size, 0) for size in region_extracted_size]):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        first_ndarray_region[block_x, block_y, block_z] = regions[0].get_block(block_x, block_y, block_z).name()
    
    for block_x, block_y, block_z in product(*[range(size) for size in region_extracted_size]):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        second_ndarray_region[block_x, block_y, block_z] = regions[1].get_block(block_x, block_y, block_z).name()
    
    extracted_edge_region = np.concatenate((first_ndarray_region, second_ndarray_region), axis=axis)
    sliced_edge_slices = [
        slice(0, axis_size)
        for axis_size in extracted_edge_region.shape
    ]
    sliced_edge_slices[axis] = slice(1, -1)
    sliced_edge_region = extracted_edge_region[sliced_edge_slices]  # cut off the edges of the region that were already previously extracted

    return sliced_edge_region

def get_corner_samples(
    file_paths: tuple[str | os.PathLike, str | os.PathLike, str | os.PathLike, str | os.PathLike],
) -> np.ndarray:  # filepaths should be in the order of top left, top right, bottom left, bottom right
    regions = []
    for file_path in file_paths:
        with open(file_path, "rb") as region_file:
            regions.append(anvil.Region.from_file(region_file))
    
    region_extracted_size = [CHUNK_LENGTH, CHUNK_HEIGHT, CHUNK_LENGTH]

    # first_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    # second_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    # third_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    # fourth_ndarray_region = np.empty(*region_extracted_size, dtype=np.uint16)
    first_ndarray_region = np.empty(*region_extracted_size, dtype=object)
    second_ndarray_region = np.empty(*region_extracted_size, dtype=object)
    third_ndarray_region = np.empty(*region_extracted_size, dtype=object)
    fourth_ndarray_region = np.empty(*region_extracted_size, dtype=object)

    for block_x, block_y, block_z in product(*[range(-size, 0) for size in region_extracted_size]):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        first_ndarray_region[block_x, block_y, block_z] = regions[0].get_block(block_x, block_y, block_z).name()
    
    for block_x, block_y, block_z in product(range(region_extracted_size[0]), range(CHUNK_HEIGHT), range(-region_extracted_size[2], 0)):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        second_ndarray_region[block_x, block_y, block_z] = regions[1].get_block(block_x, block_y, block_z).name()
    
    for block_x, block_y, block_z in product(range(-region_extracted_size[0], 0), range(CHUNK_HEIGHT), range(region_extracted_size[2])):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        third_ndarray_region[block_x, block_y, block_z] = regions[2].get_block(block_x, block_y, block_z).name()
    
    for block_x, block_y, block_z in product(range(region_extracted_size[0]), range(CHUNK_HEIGHT), range(region_extracted_size[2])):
        block_x, block_y, block_z = block_x % REGION_BLOCK_LENGTH, block_y % CHUNK_HEIGHT, block_z % REGION_BLOCK_LENGTH
        fourth_ndarray_region[block_x, block_y, block_z] = regions[3].get_block(block_x, block_y, block_z).name()
    
    extracted_corner_region = np.concatenate((
        np.concatenate((first_ndarray_region, second_ndarray_region), axis=0),
        np.concatenate((third_ndarray_region, fourth_ndarray_region), axis=0),
    ), axis=2)

    sliced_corner_region = extracted_corner_region[1:-1, :, 1:-1]  # cut off the edges of the region that were already previously extracted

    return sliced_corner_region

SAMPLES_PER_FILE = 8192

def save_samples(
    samples: list[np.ndarray],
    scores: list[float],
    output_dir: str | os.PathLike,
    min_size: int = SAMPLES_PER_FILE,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if min_size > 0:
        while len(samples) >= min_size:
            sample_tensor = th.tensor(samples[:SAMPLES_PER_FILE], dtype=th.uint8)
            exported_samples = {
                "samples": sample_tensor,
                "scores": scores[:SAMPLES_PER_FILE],
            }
            th.save(exported_samples, os.path.join(output_dir, str(uuid4()) + ".pt"))
            samples = samples[min_size:]
            scores = scores[min_size:]
    else:
        sample_tensor = th.tensor(samples, dtype=th.uint8)
        exported_samples = {
            "samples": sample_tensor,
            "scores": scores,
        }
        th.save(exported_samples, os.path.join(output_dir, str(uuid4()) + ".pt"))
        samples = []
        scores = []

    return samples, scores

_print = print
# only print from rank 0
def print(*args, **kwargs):
    if COMM.Get_rank() == 0:
        _print(*args, **kwargs)

def extract_world(
    path: str | os.PathLike,
    output_dir: str | os.PathLike = "outputs",
    intermediate_output_dir: str | os.PathLike = "intermediate_outputs",
    sample_size: tuple[int, int, int] = (16, 16, 16),
    score_threshold: float = 0.5,
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path, intermediate_output_dir)

    output_dir = os.path.join(output_dir, os.path.basename(path))
    region_dir = os.path.join(path, "region")
    available_regions, region_indices = get_available_regions(region_dir)
    extracted_samples = []
    scores = []

    rank = COMM.Get_rank()
    world_size = COMM.Get_size()
    os.makedirs(output_dir, exist_ok=True)

    for region_index_chunk in tqdm(chunked(region_indices, world_size), total=available_regions.sum() // world_size, desc="Extracting samples from regions"):
        x, z = region_index_chunk[rank]
        file_path = os.path.join(region_dir, f"r.{x}.{z}.mca")
        region_samples, region_scores = extract_from_region(file_path, sample_size, score_threshold)
        extracted_samples.extend(region_samples)
        scores.extend(region_scores)

        extracted_samples, scores = save_samples(extracted_samples, scores, output_dir)

    vertical_edge_region_samples = available_regions[:-1] & available_regions[1:]
    horizontal_edge_region_samples = available_regions[:, :-1] & available_regions[:, 1:]
    corner_region_samples = available_regions[:-1, :-1] & available_regions[:-1, 1:] & available_regions[1:, :-1] & available_regions[1:, 1:]

    print("Extracting samples from edges and corners...")

    unprocessed_samples = {}
    for sample_index_chunk in chunked(np.nonzero(vertical_edge_region_samples), world_size):
        x, z = sample_index_chunk[rank]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.mca"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}-{x + 1}.{z}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_edge_samples(file_paths, axis=0)

    for sample_index_chunk in chunked(np.nonzero(horizontal_edge_region_samples), world_size):
        x, z = sample_index_chunk[rank]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.mca"),
            os.path.join(region_dir, f"r.{x}.{z + 1}.mca"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}.{z}-{z + 1}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_edge_samples(file_paths, axis=2)

    for sample_index_chunk in chunked(np.nonzero(corner_region_samples), world_size):
        x, z = sample_index_chunk[rank]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.mca"),
            os.path.join(region_dir, f"r.{x}.{z + 1}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z + 1}.mca"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}-{x + 1}.{z}-{z + 1}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_corner_samples(file_paths)

    print("Filtering extracted edge and corner samples...")

    for unprocessed_sample in tqdm(unprocessed_samples.values(), total=len(unprocessed_samples), desc="Filtering extracted edge and corner samples"):
        region_samples, region_scores = extract_from_ndarray(unprocessed_sample, sample_size, score_threshold)
        extracted_samples.extend(region_samples)
        scores.extend(region_scores)

        extracted_samples, scores = save_samples(extracted_samples, scores, output_dir)

    # save any remaining samples
    save_samples(extracted_samples, scores, output_dir, min_size=0)

    print("Done")

def extract_zipped_world(
    path: str | os.PathLike,
    intermediate_output_dir: str | os.PathLike = "intermediate_outputs",
) -> os.PathLike:
    path_no_ext, ext = os.path.splitext(path)
    assert ext == ".zip", "File must be a zip file"

    name = os.path.basename(path_no_ext)

    world_parent_dir = os.path.join(intermediate_output_dir, name)
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(world_parent_dir)
    
    for world_dir in chain((world_parent_dir,), os.listdir(world_parent_dir)):  # only get the extracted world directory from the zip
        world_path = os.path.join(world_parent_dir, world_dir)
        if not os.path.isdir(world_path):
            continue
        
        level_dat_path = os.path.join(world_path, "level.dat")
        if os.path.exists(level_dat_path):
            return world_path
    
    raise Exception(f"No level.dat file found in {world_parent_dir} or its child directories")
