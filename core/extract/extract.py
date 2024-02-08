import os
import zipfile
import anvil
import numpy as np
import torch as th
from uuid import uuid4
from mpi4py import MPI
from itertools import chain, product
from more_itertools import chunked
from typing import Sequence


CHUNK_LENGTH = 16  # in blocks
CHUNK_HEIGHT = 256  # in blocks
REGION_LENGTH = 32  # in chunks
REGION_BLOCK_LENGTH = REGION_LENGTH * CHUNK_LENGTH  # in blocks

COMM = MPI.COMM_WORLD

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
) -> list[np.ndarray]:
    extracted_samples = []

    # extracted_samples.append(dense_array[:sample_size[0], :sample_size[1], :sample_size[2]])

    return extracted_samples

def extract_from_region(
    file_path: str | os.PathLike,
    sample_size: tuple[int, int, int],
) -> list[np.ndarray]:
    ndarray_filepath = file_path.replace(".mca", ".npy")
    if os.path.exists(ndarray_filepath):
        ndarray_blocks = np.load(ndarray_filepath)
    else:
        with open(file_path, "rb") as region_file:
            region = anvil.Region.from_file(region_file)

        # ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=np.uint16)
        ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=object)
        for x, z in product(range(REGION_LENGTH), repeat=2):
            import pdb; pdb.set_trace()
            chunk = anvil.Chunk.from_region(region, x, z)
            for block_x, block_y, block_z in product(range(x, x + CHUNK_LENGTH), range(CHUNK_HEIGHT), range(z, z + CHUNK_LENGTH)):
                ndarray_blocks[block_x, block_y, block_z] = chunk.get_block(block_x, block_y, block_z).name()

        np.save(ndarray_filepath, ndarray_blocks)

    return extract_from_ndarray(ndarray_blocks, sample_size)

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

def save_samples(
    samples: list[np.ndarray],
    output_dir: str | os.PathLike,
) -> list[np.ndarray]:
    while len(samples) >= SAMPLES_PER_FILE:
        sample_tensor = th.tensor(samples[:SAMPLES_PER_FILE], dtype=th.uint8)
        th.save(sample_tensor, os.path.join(output_dir, str(uuid4()) + ".pt"))
        samples = samples[SAMPLES_PER_FILE:]

    return samples

_print = print
# only print from rank 0
def print(*args, **kwargs):
    if COMM.Get_rank() == 0:
        _print(*args, **kwargs)

SAMPLES_PER_FILE = 2048

def extract_world(
    path: str | os.PathLike,
    sample_size: tuple[int, int, int] = (16, 16, 16),
    output_dir: str | os.PathLike = "outputs",
    intermediate_output_dir: str | os.PathLike = "intermediate_outputs",
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path, intermediate_output_dir)

    output_dir = os.path.join(output_dir, os.path.basename(path))
    region_dir = os.path.join(path, "region")
    available_regions, region_indices = get_available_regions(region_dir)
    extracted_samples = []
    rank = COMM.Get_rank()
    world_size = COMM.Get_size()
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting samples from regions...")

    for region_index_chunk in chunked(region_indices, world_size):
        x, z = region_index_chunk[rank]
        file_path = os.path.join(region_dir, f"r.{x}.{z}.mca")
        extracted_samples.extend(extract_from_region(file_path, sample_size))

        extracted_samples = save_samples(extracted_samples, output_dir)

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

    for unprocessed_sample in unprocessed_samples:
        extracted_samples.extend(extract_from_ndarray(unprocessed_sample, sample_size))

        extracted_samples = save_samples(extracted_samples, output_dir)

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
