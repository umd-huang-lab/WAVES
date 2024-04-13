import os
import sys
import click
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import dotenv
from dev import get_all_image_dir_paths

dotenv.load_dotenv(override=False)


@click.group()
def cli():
    """Watermark benchmarking tool."""
    pass


@click.command()
def version():
    """Check the version of the CLI."""
    click.echo("0.2.1")


# Worker function to run a command on a single image directory
def run_command(script_name, path, dry, args):
    cmd = (
        [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"dev_scripts/{script_name}.py",
            ),
        ]
        + list(args)
        + ["--path", path, "--quiet"]
        + (["--dry"] if dry else [])
    )
    subprocess.run(cmd)


# Common handler of main commands, supporting --all
def call_script(script_name, all, dry, args):
    # Run on a single image directory
    if not all:
        cmd = [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"dev_scripts/{script_name}.py",
            ),
        ] + list(args)
        subprocess.run(cmd)
    # Run on all image directories
    else:
        if "--path" in args or "-p" in args:
            raise ValueError(
                "Cannot use --path or -p when running on all image directories"
            )
        paths = list(
            get_all_image_dir_paths(
                lambda _dataset_name, _attack_name, _attack_strength, _source_name: ()
            ).values()
        )
        random.shuffle(paths)
        print(
            f"Running command 'wmbench {script_name}' on {len(paths)} image directories found"
        )

        if not dry:
            for path in tqdm(paths, desc="Processing", unit="dir"):
                run_command(script_name, path, False, args)
        else:
            with ProcessPoolExecutor(max_workers=16) as executor:
                # Map the function to the arguments
                futures = {
                    executor.submit(run_command, script_name, path, True, args): path
                    for path in paths
                }
                # Process the futures as they complete
                for future in tqdm(
                    as_completed(futures),
                    total=len(paths),
                    desc="Processing",
                    unit="dir",
                ):
                    path = futures[future]
                    try:
                        future.result()  # Retrieve result or exception
                    except Exception as e:
                        print(f"Error processing {path}: {e}")


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--all", "-a", is_flag=True, default=False, help="Run on all image directories"
)
@click.argument("args", nargs=-1)
def status(all, args):
    """Check and summarize the status of attacks (support --all)."""
    call_script("status", all, True, args)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--all", "-a", is_flag=True, default=False, help="Run on all image directories"
)
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.argument("args", nargs=-1)
def reverse(all, dry, args):
    """Reverse stable diffusion on attacked images (support --all)."""
    call_script("reverse", all, dry, args)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--all", "-a", is_flag=True, default=False, help="Run on all image directories"
)
@click.argument("args", nargs=-1)
def decode(all, args):
    """Decode messags from images (support --all)."""
    call_script("decode", all, False, args)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--all", "-a", is_flag=True, default=False, help="Run on all image directories"
)
@click.argument("args", nargs=-1)
def metric(all, args):
    """Measure image quality metrics (support --all)."""
    call_script("metric", all, False, args)


@click.command()
def chmod():
    """Grant group write access to all your files."""
    subprocess.run(
        [
            sys.executable,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "dev_scripts/chmod.py"
            ),
        ]
    )


@click.command()
def space():
    """[DEBUG] Start the gradio ploting interface."""
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
        ]
    )


# Add the subcommands to the main group
cli.add_command(version)
# Main commands (support --all to run on all image directories)
cli.add_command(status)
cli.add_command(reverse)
cli.add_command(decode)
cli.add_command(metric)
# Utility commands
cli.add_command(chmod)
# Debug commands
cli.add_command(space)

cli()