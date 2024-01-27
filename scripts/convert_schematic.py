import os
import numpy as np
from nbtschematic import SchematicFile


def convert_schematic(
    schematic_dir: str | os.PathLike = "raw_test_inputs",
    output_dir: str | os.PathLike = "test_inputs",
) -> None:
    for file in os.listdir(schematic_dir):
        if not file.endswith(".schematic"):
            continue

        filepath = os.path.join(schematic_dir, file)

        try:
            sf = SchematicFile.load(filepath)
        except Exception as e:
            print(f"{file} failed to load!")
            continue

        if not ("Height" in sf.root.keys() and "Length" in sf.root.keys() and "Width" in sf.root.keys()):
            print(f"{file} is incorrectly formatted!")
            continue

        y_size = int(sf.root["Height"])
        z_size = int(sf.root["Length"])
        x_size = int(sf.root["Width"])

        try:
            structure = np.copy(sf.blocks).astype(np.ubyte)
            if structure.max() > 255:
                print(f"{file} has a max block id of {structure.max()}!")
                break
        except KeyError as e:
            print(f"{file} is incorrectly formatted!")
            continue
        except ValueError as e:
            print(f"{file} block array is incorrectly formatted!")
            continue

        structure = np.transpose(structure, (1, 0, 2))  # yzx -> xyz

        print(f"{file} converted to ndarray, saving...")

        structure_path = os.path.join(output_dir, f"{file[:-10]}.npy")
        np.save(structure_path, structure)


if __name__ == "__main__":
    convert_schematic()