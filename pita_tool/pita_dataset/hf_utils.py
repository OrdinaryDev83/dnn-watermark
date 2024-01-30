"""
Utilities for interacting with HuggingFace.
"""

from dataclasses import dataclass
from typing import Dict

import click
from datasets import Dataset, load_dataset
from huggingface_hub import HfFolder, create_repo, whoami


@dataclass
class HfPitaRepository:
    repository_id: str
    commit_message: str
    dataset_directory: str
    owner: str


def set_hugging_face_token() -> None:
    """
    Set the HuggingFace token.
    """
    hf_token: str = HfFolder().get_token()

    if hf_token is not None:
        return

    hf_token: str = click.prompt(
        "Please enter your HuggingFace token (write access)", type=str, hide_input=True
    )

    HfFolder().save_token(hf_token)


def create_pita_repository(dataset_directory: str) -> HfPitaRepository:
    """
    Create a HuggingFace repository for the PITA dataset.

    Returns:
        The HuggingFace repository.
    """
    repository_id: str = click.prompt(
        "What the name of the repository to create ?",
        type=str,
        default="visible-watermark-pita",
    )

    commit_message: str = click.prompt(
        "What is the commit message?",
        type=str,
        default="generation of the pita dataset (COCO and qmul-openlogo for object detection)",
    )

    # Create the repository if it does not exist
    create_repo(
        repo_id=repository_id,
        token=HfFolder().get_token(),
        exist_ok=True,
        repo_type="dataset",
    )

    owner: Dict = whoami(token=HfFolder().get_token())

    return HfPitaRepository(
        repository_id=repository_id,
        commit_message=commit_message,
        dataset_directory=dataset_directory,
        owner=owner,
    )


def push_to_hugging_face(hf_repository: HfPitaRepository) -> None:
    """
    Push the dataset to HuggingFace Hub.

    Args:
        hf_repository: The HuggingFace repository to push to.
    """
    local_dataset: Dataset = load_dataset(hf_repository.dataset_directory)

    print(f"{hf_repository.owner['name']}/{hf_repository.repository_id}")
    local_dataset.push_to_hub(
        repo_id=f"{hf_repository.owner['name']}/{hf_repository.repository_id}",
        commit_message=hf_repository.commit_message,
        token=HfFolder().get_token(),
    )
