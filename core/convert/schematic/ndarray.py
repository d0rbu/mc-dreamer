import os
import numpy as np
from core.convert.utils import multiple
from nbtschematic import SchematicFile


@multiple
def convert(
    schematic_path: str | os.PathLike,
) -> np.ndarray:
    schematic = SchematicFile.load(schematic_path)
    
    assert "Height" in schematic.root.keys() and \
           "Length" in schematic.root.keys() and \
           "Width" in schematic.root.keys(), "Schematic is incorrectly formatted!"

    x_size = int(schematic.root['Width'])
    y_size = int(schematic.root['Height'])
    z_size = int(schematic.root['Length'])

    assert isinstance(schematic.blocks, np.ndarray), "Schematic is incorrectly formatted!"
    assert schematic.blocks.shape == (y_size, z_size, x_size), "Schematic is incorrectly formatted!"
    assert schematic.blocks.max() <= np.iinfo(np.uint8).max, "Schematic contains blocks with IDs greater than 255!"

    return schematic.blocks.transpose(2, 0, 1)  # (y, z, x) -> (x, y, z)
