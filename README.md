# Watermarking Detection - DNN


## Authors

- Bastien Pouëssel
- Arnaud Baradat
- Nicolas Fidel
- Tom Genlis
- Théo Ripoll
- Quentin Fisch

## PITA Dataset creation

The characteristics of the dataset we generated are the following:
1. Single watermarked images
3. Aligned bounding boxes (center, width, height)
4. 512x512 images
5. Base images from coco (all classes)
6. Logos from this dataset: https://hangsu0730.github.io/qmul-openlogo/

The dataset contains subdatasets for:
- Text watermarks
- Logo watermarks

We worked on multiple properties of the watermarks:
- Color
- Size
- Position
- Opacity
- Font

There two supported formats for the dataset:
- COCO
- YOLO (Ultralytics)

Everything related to the PITA dataset is located in the `pita_tool` folder. We created a CLI tool to generate and download the dataset. You can find help about the CLI tool by simply running the following command:
```bash
python pita_tool/pita.py
```

The dataset is hosted on Hugging Face and is available at the following link: https://huggingface.co/datasets/bastienp/visible-watermark-pita

## Model

The following models have been fine-tuned on our dataset, and tested on CLWD dataset (https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view)
- [X] YoloV8 Nano
- [X] YoloV8 Large
- [X] DETR w/ ResNet backbone
- [X] Faster R-CNN

PITA fine-tuned YoloV8 Nano model is available on Hugging Face too at: https://huggingface.co/qfisch/yolov8n-watermark-detection

## Results

You can find the results and the paper explaining our work in the `report` folder.
A demo using our fine-tuned YoloV8 Nano model is available on a Hugging Face Space here: https://huggingface.co/spaces/qfisch/watermark-detection

Note: We were enable to correctly benchmark DETR, thus we decided to not include any result for this model.


### Weights & Biases

- Faster R-CNN training: https://api.wandb.ai/links/qfisch/g7jzfptj
- YoloV8 (Large, similar for nano) training: https://api.wandb.ai/links/qfisch/ni8916bg
