import os
from core.extract.extract import extract_zipped_world


def extract_data(
    output_dir: str | os.PathLike = "outputs",
    raw_output_dir: str | os.PathLike = "raw_outputs",
) -> None:
    for file in os.listdir(raw_output_dir):
        file_path = os.path.join(raw_output_dir, file)
        if not os.path.isfile(file_path):
            continue
        
        if file.endswith(".zip") or os.path.isdir(file_path):
            extract_zipped_world(file_path, output_dir)
        else:
            print(f"Unknown file type: {file}")


if __name__ == "__main__":
    extract_data()
