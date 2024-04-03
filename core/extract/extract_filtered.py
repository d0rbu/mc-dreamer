import os
import anvil
import numpy as np
import torch as th
from tqdm import tqdm
from mpi4py import MPI
from itertools import product
from more_itertools import chunked
from typing import Sequence
from core.extract.extract import extract_zipped_world, get_available_regions, sample_filename, print, save_samples, REGION_BLOCK_LENGTH, REGION_LENGTH, CHUNK_HEIGHT, CHUNK_LENGTH, BLOCK_MAP
from core.extract.filters.filter import Filter


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

def extract_world(
    path: str | os.PathLike,
    output_dir: str | os.PathLike = "outputs",
    intermediate_output_dir: str | os.PathLike = "intermediate_outputs",
    filters: Sequence[Filter] = [],
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path, intermediate_output_dir)

    output_dir = os.path.join(output_dir, os.path.basename(path))
    region_dir = os.path.join(path, "region")
    available_regions, region_indices = get_available_regions(region_dir)

    world_size = COMM.Get_size()
    os.makedirs(output_dir, exist_ok=True)

    region_index_chunk_generator = chunked(region_indices, world_size)

    if RANK == 0:
        region_index_chunk_generator = tqdm(region_index_chunk_generator, total=available_regions.sum() // world_size, leave=False, desc="Extracting samples from regions")

    for region_index_chunk in region_index_chunk_generator:
        if len(region_index_chunk) <= RANK:  # no more work left for us this iteration!
            break

        x, z = region_index_chunk[RANK]
        region_name = f"r.{x}.{z}"
        filename = f"{region_name}.pt"

        if os.path.exists(os.path.join(output_dir, filename)):  # skip already processed regions
            continue

        file_path = os.path.join(region_dir, f"{region_name}.mca")

        if os.path.getsize(file_path) == 0:  # skip empty regions
            continue

        tensor_blocks = region_to_tensor(file_path)
        current_mask = None

        for filter in filters:
            new_mask = filter(tensor_blocks, current_mask)

            tensor_blocks, current_mask = shrink_blocks(tensor_blocks, new_mask)

        save_samples(tensor_blocks, current_mask, output_dir, filename)

    print(f"{RANK} done!")

def shrink_blocks(
    tensor_blocks: th.Tensor,
    filter: th.Tensor | None,
) -> th.Tensor:
    # squeeze dimensions if, for example, the entire top layer is 0 in the filter
    return tensor_blocks, filter

def region_to_tensor(
    file_path: str | os.PathLike,
) -> th.Tensor:
    tensor_filepath = file_path.replace(".mca", ".pt")
    if os.path.exists(tensor_filepath):
        tensor_blocks = th.load(tensor_filepath)
    else:
        with open(file_path, "rb") as region_file:
            region = anvil.Region.from_file(region_file)

        tensor_blocks = th.empty((REGION_BLOCK_LENGTH, CHUNK_HEIGHT, REGION_BLOCK_LENGTH), dtype=th.uint8)
        for chunk_x, chunk_z in tqdm(product(range(REGION_LENGTH), repeat=2), total=REGION_LENGTH ** 2, leave=False, desc=f"Converting region {os.path.basename(file_path)} to tensor"):
            try:
                chunk = anvil.Chunk.from_region(region, chunk_x, chunk_z)
            except anvil.errors.ChunkNotFound as e:
                chunk = anvil.EmptyChunk(chunk_x, chunk_z)
            for block_x, block_y, block_z in product(range(CHUNK_LENGTH), range(CHUNK_HEIGHT), range(CHUNK_LENGTH)):
                x, y, z = chunk_x * CHUNK_LENGTH + block_x, block_y, chunk_z * CHUNK_LENGTH + block_z
                block_name = chunk.get_block(block_x, block_y, block_z).name()
                tensor_blocks[x, y, z] = BLOCK_MAP[block_name]

        th.save(tensor_blocks, tensor_filepath)

    return tensor_blocks

def save_samples(
    tensor_blocks: th.Tensor,
    filter: th.Tensor,
    output_dir: str | os.PathLike,
    filename: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    joined_samples = {
        "region": th.tensor(tensor_blocks, dtype=th.uint8),
        "filter": filter,
    }

    th.save(joined_samples, os.path.join(output_dir, filename))
