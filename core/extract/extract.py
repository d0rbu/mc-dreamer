import os
import zipfile
import asyncio
import anvil
import numpy as np
from functools import partial
from itertools import chain, product
from typing import Sequence


CHUNK_LENGTH = 16  # in blocks
CHUNK_HEIGHT = 256  # in blocks
REGION_LENGTH = 32  # in chunks
REGION_BLOCK_LENGTH = REGION_LENGTH * CHUNK_LENGTH  # in blocks

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

def collate_and_save_samples(
    output_dir: str | os.PathLike,
    samples_future: asyncio.Future,
) -> np.ndarray:
    all_samples = samples_future.result()
    samples = np.concatenate(all_samples, axis=0)
    np.save(output_dir, samples)

    return samples

def extract_from_ndarray(
    dense_array: np.ndarray,
) -> np.ndarray:
    extracted_samples = []
    
    return extracted_samples

def extract_from_region(
    file_path: str | os.PathLike,
    future: asyncio.Future,
) -> None:
    with open(file_path, "rb") as region_file:
        region = anvil.Region.from_file(region_file)

    # ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=np.uint16)
    ndarray_blocks = np.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=object)
    for x, z in product(range(REGION_LENGTH), repeat=2):
        chunk = anvil.Chunk.from_region(region, (x, z))
        for block_x, block_y, block_z in product(range(x, x + CHUNK_LENGTH), range(CHUNK_HEIGHT), range(z, z + CHUNK_LENGTH)):
            ndarray_blocks[block_x, block_y, block_z] = chunk.get_block(block_x, block_y, block_z).name()
    
    future.set_result(extract_from_ndarray(ndarray_blocks))

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

async def extract_world_async(
    path: str | os.PathLike,
    output_dir: str | os.PathLike = "outputs",
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path)

    output_path = os.path.join(output_dir, os.path.basename(path))
    region_dir = os.path.join(path, "region")
    available_regions, region_indices = get_available_regions(region_dir)
    extracted_sample_futures = []

    for x, z in region_indices:
        file_path = os.path.join(region_dir, f"r.{x}.{z}.mca")
        future = asyncio.Future()
        extracted_sample_futures.append(future)
        asyncio.create_task(asyncio.to_thread(extract_from_region, file_path, future))
    
    process_samples_callback = partial(collate_and_save_samples, output_path)
    
    get_all_samples_future = asyncio.gather(*extracted_sample_futures)
    get_all_samples_future.add_done_callback(process_samples_callback)

    await get_all_samples_future
    
    vertical_edge_region_samples = available_regions[:-1] & available_regions[1:]
    horizontal_edge_region_samples = available_regions[:, :-1] & available_regions[:, 1:]
    corner_region_samples = available_regions[:-1, :-1] & available_regions[:-1, 1:] & available_regions[1:, :-1] & available_regions[1:, 1:]

    unprocessed_samples = []
    for x, z in chain(np.nonzero(vertical_edge_region_samples), np.nonzero(horizontal_edge_region_samples)):
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.mca"),
        )
        unprocessed_samples.append(get_edge_samples(file_paths, axis=0))
    
    for x, z in np.nonzero(corner_region_samples):
        file_paths = (
            os.path.join(region_dir, f"r.{x}.{z}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z}.mca"),
            os.path.join(region_dir, f"r.{x}.{z + 1}.mca"),
            os.path.join(region_dir, f"r.{x + 1}.{z + 1}.mca"),
        )
        unprocessed_samples.append(get_corner_samples(file_paths))
    
    extracted_sample_futures = []
    for unprocessed_sample in unprocessed_samples:
        future = asyncio.Future()
        extracted_sample_futures.append(future)
        # TODO: process the samples

def extract_world(
    path: str | os.PathLike,
    output_dir: str | os.PathLike = "outputs",
) -> None:
    asyncio.run(extract_world_async(path, output_dir))

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
