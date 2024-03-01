import os
import zipfile
import json
import pickle
import anvil
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
) -> np.ndarray:
    scores = np.empty((dense_array.shape[0] - sample_size[0] + 1, dense_array.shape[1] - sample_size[1] + 1, dense_array.shape[2] - sample_size[2] + 1), dtype=np.float32)
    running_integral = np.zeros((sample_size[0] + 1, *dense_array.shape[1:], NUM_BLOCKS), dtype=np.uintc)

    x_generator = range(dense_array.shape[0])
    if COMM.Get_rank() == 0:
        x_generator = tqdm(x_generator, total=dense_array.shape[0], leave=False, desc="Scoring samples")

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

    return scores

def extract_from_region(
    file_path: str | os.PathLike,
    sample_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ndarray_filepath = file_path.replace(".mca", ".npy")
    if os.path.exists(ndarray_filepath):
        ndarray_blocks = np.load(ndarray_filepath)
    else:
        with open(file_path, "rb") as region_file:
            region = anvil.Region.from_file(region_file)

        ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=np.ubyte)
        for chunk_x, chunk_z in tqdm(product(range(REGION_LENGTH), repeat=2), total=REGION_LENGTH ** 2, leave=False, desc=f"Converting region {os.path.basename(file_path)} to ndarray"):
            try:
                chunk = anvil.Chunk.from_region(region, chunk_x, chunk_z)
            except anvil.errors.ChunkNotFound as e:
                chunk = anvil.EmptyChunk(chunk_x, chunk_z)
            for block_x, block_y, block_z in product(range(CHUNK_LENGTH), range(CHUNK_HEIGHT), range(CHUNK_LENGTH)):
                x, y, z = chunk_x * CHUNK_LENGTH + block_x, block_y, chunk_z * CHUNK_LENGTH + block_z
                block_name = chunk.get_block(block_x, block_y, block_z).name()
                ndarray_blocks[x, y, z] = BLOCK_MAP[block_name]

        np.save(ndarray_filepath, ndarray_blocks)

    return ndarray_blocks, extract_from_ndarray(ndarray_blocks, sample_size)

def get_edge_samples(
    file_paths: tuple[str | os.PathLike, str | os.PathLike],
    axis: int = 0,  # which axis to stitch the samples along, should be 0 or 2
) -> np.ndarray:
    region_ndarrays = []
    for file_path in file_paths:
        region_ndarrays.append(np.load(file_path))

    region_extracted_size = [REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH]
    region_extracted_size[axis] = CHUNK_LENGTH - 1
    
    first_ndarray_region = np.empty(region_extracted_size, dtype=np.ubyte)
    second_ndarray_region = np.empty(region_extracted_size, dtype=np.ubyte)

    for block_x, block_y, block_z in product(*[range(-size, 0) for size in region_extracted_size]):
        first_ndarray_region[block_x, block_y, block_z] = region_ndarrays[0][block_x, block_y, block_z]
    
    for block_x, block_y, block_z in product(*[range(size) for size in region_extracted_size]):
        second_ndarray_region[block_x, block_y, block_z] = region_ndarrays[1][block_x, block_y, block_z]
    
    extracted_edge_region = np.concatenate((first_ndarray_region, second_ndarray_region), axis=axis)

    return extracted_edge_region

def get_corner_samples(
    file_paths: tuple[str | os.PathLike, str | os.PathLike, str | os.PathLike, str | os.PathLike],
) -> np.ndarray:  # filepaths should be in the order of top left, top right, bottom left, bottom right
    region_ndarrays = []
    for file_path in file_paths:
        region_ndarrays.append(np.load(file_path))

    region_extracted_size = [CHUNK_LENGTH - 1, CHUNK_HEIGHT, CHUNK_LENGTH - 1]

    first_ndarray_region = np.empty(region_extracted_size, dtype=np.uint16)
    second_ndarray_region = np.empty(region_extracted_size, dtype=np.uint16)
    third_ndarray_region = np.empty(region_extracted_size, dtype=np.uint16)
    fourth_ndarray_region = np.empty(region_extracted_size, dtype=np.uint16)

    for block_x, block_y, block_z in product(*[range(-size, 0) for size in region_extracted_size]):
        first_ndarray_region[block_x, block_y, block_z] = region_ndarrays[0][block_x, block_y, block_z]

    for block_x, block_y, block_z in product(range(region_extracted_size[0]), range(CHUNK_HEIGHT), range(-region_extracted_size[2], 0)):
        second_ndarray_region[block_x, block_y, block_z] = region_ndarrays[1][block_x, block_y, block_z]

    for block_x, block_y, block_z in product(range(-region_extracted_size[0], 0), range(CHUNK_HEIGHT), range(region_extracted_size[2])):
        third_ndarray_region[block_x, block_y, block_z] = region_ndarrays[2][block_x, block_y, block_z]

    for block_x, block_y, block_z in product(range(region_extracted_size[0]), range(CHUNK_HEIGHT), range(region_extracted_size[2])):
        fourth_ndarray_region[block_x, block_y, block_z] = region_ndarrays[3][block_x, block_y, block_z]

    extracted_corner_region = np.concatenate((
        np.concatenate((first_ndarray_region, second_ndarray_region), axis=0),
        np.concatenate((third_ndarray_region, fourth_ndarray_region), axis=0),
    ), axis=2)

    return extracted_corner_region

def sample_filename(sample_name: str, sample_size: tuple[int, int, int]) -> str:
    size_str = "x".join(map(str, sample_size))
    return f"{size_str}_{sample_name}.pt"

def save_samples(
    ndarray_region: np.ndarray,
    scores: np.ndarray,
    output_dir: str | os.PathLike,
    sample_name: str,
    sample_size: tuple[int, int, int] = (16, 16, 16),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    joined_samples = {
        "region": th.tensor(ndarray_region, dtype=th.uint8),
        "scores": th.tensor(scores, dtype=th.float32),
    }

    filename = sample_filename(sample_name, sample_size)
    th.save(joined_samples, os.path.join(output_dir, filename))

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
    extract_edge_and_corner_samples: bool = False,
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path, intermediate_output_dir)

    output_dir = os.path.join(output_dir, os.path.basename(path))
    region_dir = os.path.join(path, "region")
    available_regions, region_indices = get_available_regions(region_dir)

    rank = COMM.Get_rank()
    world_size = COMM.Get_size()
    os.makedirs(output_dir, exist_ok=True)

    region_index_chunk_generator = chunked(region_indices, world_size)

    if COMM.Get_rank() == 0:
        region_index_chunk_generator = tqdm(region_index_chunk_generator, total=available_regions.sum() // world_size, leave=False, desc="Extracting samples from regions")

    for region_index_chunk in region_index_chunk_generator:
        if len(region_index_chunk) <= rank:  # no more work left for us this iteration!
            break

        x, z = region_index_chunk[rank]
        region_name = f"r.{x}.{z}"
        filename = sample_filename(region_name, sample_size)

        if os.path.exists(os.path.join(output_dir, filename)):  # skip already processed regions
            continue

        file_path = os.path.join(region_dir, f"{region_name}.mca")

        if os.path.getsize(file_path) == 0:  # skip empty regions
            continue

        ndarray_region, scores = extract_from_region(file_path, sample_size)
        save_samples(ndarray_region, scores, output_dir, region_name, sample_size)
    
    if not extract_edge_and_corner_samples:
        print("Done!")
        return

    vertical_edge_region_samples = available_regions[:-1] & available_regions[1:]
    horizontal_edge_region_samples = available_regions[:, :-1] & available_regions[:, 1:]
    corner_region_samples = available_regions[:-1, :-1] & available_regions[:-1, 1:] & available_regions[1:, :-1] & available_regions[1:, 1:]

    print("Extracting samples from edges and corners...")

    unprocessed_samples = {}
    for sample_index_chunk in chunked(np.nonzero(vertical_edge_region_samples), world_size):
        if len(sample_index_chunk) <= rank:  # no more work left for us this iteration!
            break

        x = sample_index_chunk[rank][0]
        z = sample_index_chunk[rank][1]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.npy"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.npy"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}-{x + 1}.{z}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_edge_samples(file_paths, axis=0)
            np.save(new_file_path, unprocessed_samples[new_file_path])

    for sample_index_chunk in chunked(np.nonzero(horizontal_edge_region_samples), world_size):
        if len(sample_index_chunk) <= rank:  # no more work left for us this iteration!
            break

        x = sample_index_chunk[rank][0]
        z = sample_index_chunk[rank][1]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.npy"),
            os.path.join(region_dir, f"r.{x}.{z + 1}.npy"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}.{z}-{z + 1}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_edge_samples(file_paths, axis=2)
            np.save(new_file_path, unprocessed_samples[new_file_path])

    for sample_index_chunk in chunked(np.nonzero(corner_region_samples), world_size):
        if len(sample_index_chunk) <= rank:  # no more work left for us this iteration!
            break

        x = sample_index_chunk[rank][0]
        z = sample_index_chunk[rank][1]
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.npy"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.npy"),
            os.path.join(region_dir, f"r.{x}.{z + 1}.npy"),
            os.path.join(region_dir, f"r.{x + 1}.{z + 1}.npy"),
        )
        new_file_path = os.path.join(region_dir, f"r.{x}-{x + 1}.{z}-{z + 1}.npy")
        if os.path.exists(new_file_path):
            unprocessed_samples[new_file_path] = np.load(new_file_path)
        else:
            unprocessed_samples[new_file_path] = get_corner_samples(file_paths)
            np.save(new_file_path, unprocessed_samples[new_file_path])

    print("Filtering extracted edge and corner samples...")

    for unprocessed_file_path, unprocessed_sample in tqdm(unprocessed_samples.items(), total=len(unprocessed_samples), leave=False, desc="Extracting edge and corner samples"):
        unprocessed_region_name = os.path.basename(unprocessed_file_path).replace(".npy", "")
        filename = sample_filename(unprocessed_region_name, sample_size)

        if os.path.exists(os.path.join(output_dir, filename)):  # skip already processed sections
            continue

        scores = extract_from_ndarray(unprocessed_sample, sample_size)

        save_samples(unprocessed_sample, scores, output_dir, unprocessed_region_name, sample_size)

def extract_zipped_world(
    path: str | os.PathLike,
    intermediate_output_dir: str | os.PathLike = "intermediate_outputs",
) -> os.PathLike:
    path_no_ext, ext = os.path.splitext(path)
    assert ext == ".zip", "File must be a zip file"

    name = os.path.basename(path_no_ext)

    world_parent_dir = os.path.join(intermediate_output_dir, name)
    os.makedirs(world_parent_dir, exist_ok=True)
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
