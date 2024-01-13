import os
import zipfile
from itertools import chain


def extract_world(
    path: str | os.PathLike,
    output_dir: str | os.PathLike = "outputs",
) -> None:
    if os.path.isfile(path):
        path = extract_zipped_world(path)

    output_path = os.path.join(output_dir, os.path.basename(path))


def extract_zipped_world(
    path: str | os.PathLike,
    raw_output_dir: str | os.PathLike = "raw_outputs",
) -> os.PathLike:
    path_no_ext, ext = os.path.splitext(path)
    assert ext == ".zip", "File must be a zip file"

    name = os.path.basename(path_no_ext)

    world_parent_dir = os.path.join(raw_output_dir, name)
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(world_parent_dir)
    
    for world_dir in chain((world_parent_dir,), os.listdir(world_dir)):  # only get the extracted world directory
        world_path = os.path.join(world_parent_dir, world_dir)
        if not os.path.isdir(world_path):
            continue
        
        level_dat_path = os.path.join(world_path, "level.dat")
        if os.path.exists(level_dat_path):
            return world_path
    
    raise Exception(f"No level.dat file found in {world_parent_dir} or its child directories")
