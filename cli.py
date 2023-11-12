import os
import sys
import click
import subprocess


@click.group()
def cli():
    """Watermark benchmarking tool."""
    pass


@click.command()
def reverse():
    """Reverse stable diffusion on attacked images. Run this command inside the image directory."""
    print("I am here", os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "dev_scripts/reverse.py"
            ),
        ]
    )


# Add the subcommands to the main group
cli.add_command(reverse)
