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


def download_from_hub(split_name: str, data_dir: str, format: str) -> None:
    """
    Download the dataset from HuggingFace Hub.

    Args:
        split_name (str): The split name of the dataset.
        data_dir (str): The directory of the dataset.
        format (str): The format of the dataset.
    """

    # Download the dataset
    train_dataset = f"https://huggingface.co/datasets/bastienp/visible-watermark-pita/resolve/main/data/{split_name}.zip?download=true"
    zip_split_directory = data_dir + f"/{split_name}.zip"
    wget.download(train_dataset, out=zip_split_directory)

    if format == "yolo" and not os.path.exists("metadata.yml"):
        wget.download(
            "https://huggingface.co/datasets/bastienp/visible-watermark-pita/resolve/main/data/metadata.yml?download=true",
            out=f"{data_dir}/metadata.yml",
        )

    # Extract the dataset split
    with zipfile.ZipFile(zip_split_directory, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_split_directory)


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
@click.option(
    "--format", "-f", required=True, help="The format of the dataset.", type=str
)
def download(split: str, data_dir: str, format: str) -> None:
    """Download the pita dataset from HuggingFace Datasets without generating it."""

    click.echo("Downloading the pita dataset from HuggingFace storage...")

    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if split not in ["train", "validation", "test", "all"]:
        click.echo(f"The split {split} is not valid.")
        return

    if format == "coco":
        huffing_face_split = split
    elif format == "yolo":
        huffing_face_split = split + "-yolo"
    else:
        click.echo(f"The format {format} is not valid.")
        return

    if split == "all":
        for split_name in ["train", "val", "test"]:
            download_from_hub(split_name=split_name, data_dir=data_dir, format=format)
    else:
        download_from_hub(split_name=huffing_face_split, data_dir=data_dir, format=format)
    


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
@click.option(
    "-f", "--format", default="coco", help="The format of the dataset.", type=str
)
def generate(
    dataset_directory: str, split: str, size: int, push_to_hub: bool, format: str
) -> None:
    """Generate the pita dataset from COCO and logos from QMUL-OpenLogo."""
    click.echo("Generating the pita dataset ...")

    # Create the dataset directory if it does not exist
    if push_to_hub:
        set_hugging_face_token()
        hf_repository: HfPitaRepository = create_pita_repository(
            dataset_directory=dataset_directory
        )

    # Dataset configuration
    pita_dataset: PitaDataset = PitaDataset(
        dataset_directory=dataset_directory,
        split=split,
        size=size,
        dataset_format=format,
    )

    generate_dataset(dataset=pita_dataset)

    if push_to_hub:
        push_to_hugging_face(hf_repository=hf_repository)


if __name__ == "__main__":
    pita()
