from pathlib import Path

import typer
from ruamel.yaml import YAML


def main(
    failed_ids_file: Path = Path("results_failed_motion_tracker_smpl/failed_motions_amass_train.txt"),
    input_yaml: Path = Path("data/yaml_files/amass_train.yaml"),
    output_yaml: Path = Path("data/yaml_files/amass_train_difficult_125.yaml"),
):
    """Create a difficult-only motion YAML from failed motion IDs."""
    failed_ids = []
    with open(failed_ids_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                failed_ids.append(int(line))
    failed_indices = set(failed_ids)
    print("num_failed_indices:", len(failed_indices))

    yaml = YAML()
    with open(input_yaml, "r") as f:
        data = yaml.load(f)

    filtered_motions = [m for m in data["motions"] if m["idx"] in failed_indices]
    for new_idx, motion in enumerate(filtered_motions):
        motion["idx"] = new_idx
    data["motions"] = filtered_motions
    print("num_filtered_motions:", len(data["motions"]))

    with open(output_yaml, "w") as f:
        yaml.dump(data, f)
    print(f"saved: {output_yaml}")


if __name__ == "__main__":
    typer.run(main)
