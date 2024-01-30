"""
A CLI for the to generate and download pita dataset watermark object detection dataset.
"""

import os
import zipfile

import click
import wget

from pita_dataset import (
    HfPitaRepository,
    PitaDataset,
    create_pita_repository,
    generate_dataset,
    push_to_hugging_face,
    set_hugging_face_token,
)


@click.group()
def pita() -> None:
    """A CLI for the to generata and download pita dataset watermark object detection dataset."""
    pass


@pita.command()
@click.option(
    "--split", "-s", default="train", help="The split of the dataset.", type=str
)
@click.option(
    "--data_dir",
    "-d",
    default="output_dataset",
    help="The directory of the dataset.",
    type=str,
)
def download(split: str, data_dir: str) -> None:
    """Download the pita dataset from HuggingFace Datasets without generating it."""

    click.echo("Downloading the pita dataset from HuggingFace storage...")

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if the split is valid
    if split not in ["train", "validation", "test"]:
        click.echo(f"The split {split} is not valid.")
        return

    # Download the dataset
    train_dataset = f"https://huggingface.co/datasets/bastienp/visible-watermark-pita/resolve/main/data/{split}.zip?download=true"
    zip_split_directory = data_dir + f"/{split}.zip"
    split_directory = data_dir + f"/{split}"
    wget.download(train_dataset, out=zip_split_directory)

    # Extract the dataset split
    with zipfile.ZipFile(zip_split_directory, "r") as zip_ref:
        zip_ref.extractall(split_directory)

    os.remove(zip_split_directory)


@pita.command()
@click.option(
    "--dataset_directory",
    default="data",
    help="The directory of the dataset.",
    type=str,
)
@click.option("--split", default="train", help="The split of the dataset.", type=str)
@click.option("-s", "--size", default=20_000, help="The size of the dataset.", type=int)
@click.option(
    "-p",
    "--push_to_hub",
    default=False,
    is_flag=True,
    help="Push the dataset to HuggingFace.",
)
def generate(dataset_directory: str, split: str, size: int, push_to_hub: bool) -> None:
    """Generate the pita dataset from COCO and logos from QMUL-OpenLogo."""
    click.echo("Generating the pita dataset ...")

    # Create the dataset directory if it does not exist
    if push_to_hub:
        set_hugging_face_token()
        hf_repository = create_pita_repository(
            dataset_directory=dataset_directory
        )

    # Dataset configuration
    pita_dataset: PitaDataset = PitaDataset(
        dataset_directory=dataset_directory,
        split=split,
        size=size,
    )

    generate_dataset(dataset=pita_dataset)

    if push_to_hub:
        push_to_hugging_face(hf_repository=hf_repository)


if __name__ == "__main__":
    pita()
