import os
from core.heuristics.ref_scores import (
    set_ref_score,
)


def add_heuristic_reference_score(
    reference_input_dir: str | os.PathLike = "test_inputs",
) -> None:
    name = input("Enter name: ")
    confirmed = False
    while not confirmed:
        print("Score each reference file from 0 to 10.\n")

        score = {}

        for file in os.listdir(reference_input_dir):
            if not file.endswith(".npy"):
                continue

            file_score = -1
            while file_score < 0 or file_score > 10:
                try:
                    file_score = float(input(f"Score for {file} (0-10): "))
                except ValueError:
                    print("Invalid score.")
        
            score[file] = file_score

        print("\nScored files:")
        for file, file_score in score.items():
            print(f"{file}: {file_score}")

        confirmed = input("\nConfirm? (y/n): ").lower() == "y"

    set_ref_score(name, score)

    print(f"Thanks {name}, your scores have been added.")


if __name__ == "__main__":
    add_heuristic_reference_score()
