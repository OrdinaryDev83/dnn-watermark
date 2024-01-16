from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="data/dataset", split="train")
print(dataset)
#dataset.push_to_hub("stevhliu/my-image-captioning-dataset")