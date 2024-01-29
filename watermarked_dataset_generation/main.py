import click
import os
from dataset_generation import generate_dataset, download_annotations

@click.command()
@click.option('--annotations_url', default='http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip', help='The url of the annotations file.')
@click.option('--dataset_directory', default='data', help='The directory of the dataset.')
@click.option('--split', default='train', help='The split of the dataset.')
@click.option('--annotations_file', default='data/annotations/image_info_unlabeled2017.json', help='The annotations file.')
@click.option('--output_file', default='metadata.jsonl', help='The output file.')
@click.option('--size', default=20_000, help='The size of the dataset.')
def build_dataset(annotations_url: str, dataset_directory: str, split: str, annotations_file: str, output_file: str, size: int) -> None:
    
    # check if annotations file exists
    if not os.path.exists(annotations_file):
        download_annotations(annotions_url=annotations_url, output_directory=dataset_directory)

    generate_dataset(dataset_directory + "/" + split, annotations_file, output_file, size)

if __name__ == "__main__":
    build_dataset()
