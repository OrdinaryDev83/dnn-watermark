"""
A CLI for the to generate and download pita dataset watermark object detection dataset.
"""

import os
import zipfile

import click
import wget
from datasets import load_dataset
from huggingface_hub import HfFolder

from dataset_generation import PitaDataset, generate_dataset


@click.group()
def pita():
    """A CLI for the to generata and download pita dataset watermark object detection dataset."""
    pass


@pita.command()
@click.option(
    "--split", "-s", default="train", help="The split of the dataset.", type=str
)
@click.option(
    "--data_dir",
    "-d",
    default="pita_dataset",
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
        raise ValueError("The split must be one of 'train', 'validation' or 'test'.")

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
@click.option("--size", default=20_000, help="The size of the dataset.", type=int)
@click.option("--hf", default=False, help="Push the dataset to HuggingFace.", type=bool)
def generate(dataset_directory: str, split: str, size: int, hf: bool) -> None:
    """Generate the pita dataset from COCO and logos from QMUL-OpenLogo."""
    click.echo("Generating the pita dataset ...")

    if hf:
        hf_token = click.prompt("Please enter your HuggingFace token", hide_input=True, type=str)
        HfFolder.save_token(hf_token)

    # Create the data directory if it does not exist
    pita_dataset: PitaDataset = PitaDataset(
        dataset_directory=dataset_directory,
        split=split,
        size=size,
    )

    generate_dataset(pita_dataset)

    pita_dataset.zip_dataset()
        


if __name__ == "__main__":
    pita()
