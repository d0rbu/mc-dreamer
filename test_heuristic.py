import os
import yaml
import numpy as np
from core.heuristics import HEURISTICS


def pprint_results(
    results: list[dict[str, float]],
) -> None:
    longest_heuristic_name = max(len(heuristic.__name__) for heuristic in HEURISTICS)
    
    for heuristic in HEURISTICS:
        print(f"{heuristic.__name__}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")
    
    print("\b\b ")
    
    for result in results:
        for heuristic in HEURISTICS:
            print(f"{result[heuristic.__name__]:.4f}{' ' * (longest_heuristic_name - len(heuristic.__name__))} | ", end="")
        
        print("\b\b ")


def test_heuristics(
    output_dir: str | os.PathLike = "test_outputs",
    input_dir: str | os.PathLike = "test_inputs",
) -> None:
    files = [file for file in os.listdir(input_dir) if file.endswith(".npy")]
    results = {
        file: {}
        for file in files
    }

    for file in files:
        file_path = os.path.join(input_dir, file)
        np_sample = np.load(file_path)

        for heuristic in HEURISTICS:
            results[file][heuristic.__name__] = heuristic(np_sample)

    yaml.dump(results, open(os.path.join(output_dir, "heuristics_results.yaml"), "w"))
    pprint_results(results)


if __name__ == "__main__":
    test_heuristics()
