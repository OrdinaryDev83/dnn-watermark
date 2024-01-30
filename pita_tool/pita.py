
import click
from dataset_generation.dataset import PitaDataset
from dataset_generation.generate_dataset import generate_dataset

# @click.command()
# @click.option(
#     "--annotations_url",
#     default="http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
#     help="The url of the annotations file.",
# )
# @click.option(
#     "--dataset_directory", default="data", help="The directory of the dataset."
# )
# @click.option("--split", default="train", help="The split of the dataset.")
# @click.option(
#     "--annotations_file",
#     default="data/annotations/image_info_unlabeled2017.json",
#     help="The annotations file.",
# )
# @click.option("--output_file", default="metadata.jsonl", help="The output file.")
# @click.option("--size", default=20_000, help="The size of the dataset.")
# def build_dataset(
#     annotations_url: str,
#     dataset_directory: str,
#     split: str,
#     annotations_file: str,
#     output_file: str,
#     size: int,
# ) -> None:
#     # check if annotations file exists
#     if not os.path.exists(annotations_file):
#         download_annotations(
#             annotions_url=annotations_url, output_directory=dataset_directory
#         )

#     generate_dataset(
#         dataset_directory + "/" + split, annotations_file, output_file, size
#     )


@click.group()
def pita():
    pass


@pita.command()
def download() -> None:
    click.echo("Downloading the pita dataset ...")

    pita_dataset : PitaDataset = PitaDataset(
        dataset_directory="data",
        split="train",
        size=2,
    )

    # generate_dataset(
    #     dataset_directory="data/train", annotations_file="data/annotations/image_info_unlabeled2017.json", output_file="metadata.jsonl", size=10
    # )

    generate_dataset(pita_dataset)

    pita_dataset.zip_dataset()


@pita.command()
def generate():
    click.echo("Dropped the database")


if __name__ == "__main__":
    pita()
