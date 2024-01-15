import os
import zipfile
import asyncio
import anvil
import numpy as np
from functools import partial
from itertools import chain
from typing import Sequence


CHUNK_SIZE = (16, 16, 256)  # in blocks
REGION_SIZE = (32, 32)  # in chunks

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
    
    # ndarray_blocks = np.empty((REGION_SIZE[0] * CHUNK_SIZE[0], REGION_SIZE[1] * CHUNK_SIZE[1], CHUNK_SIZE[2]), dtype=np.uint16)
    ndarray_blocks = np.empty((REGION_SIZE[0] * CHUNK_SIZE[0], REGION_SIZE[1] * CHUNK_SIZE[1], CHUNK_SIZE[2]), dtype=object)
    for x in range(REGION_SIZE[0]):
        for z in range(REGION_SIZE[1]):
            chunk = anvil.Chunk.from_region(region, (x, z))
            for block_x in range(x, x + CHUNK_SIZE[0]):
                for block_z in range(z, z + CHUNK_SIZE[1]):
                    for block_y in range(CHUNK_SIZE[2]):
                        ndarray_blocks[block_x, block_z, block_y] = chunk.get_block(block_x, block_y, block_z).name()
    
    future.set_result(extract_from_ndarray(ndarray_blocks))

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

    samples = await get_all_samples_future
    
    vertical_edge_samples

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
