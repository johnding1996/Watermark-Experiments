import os
import click
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    QUALITY_METRICS,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
)


@click.command()
@click.option(
    "--path", "-p", type=str, default=os.getcwd(), help="Path to image directory"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(path, dry, subset, quiet, limit=LIMIT, subset_limit=SUBSET_LIMIT):
    (
        dataset_name,
        attack_name,
        attack_strength,
        source_name,
    ) = parse_image_dir_path(path, quiet=quiet)
    # Metric is only required for watermarked images
    if attack_name is None or attack_strength is None:
        if not quiet:
            print(f"Metric is only required for watermarked images, exiting")
        return


if __name__ == "__main__":
    main()
