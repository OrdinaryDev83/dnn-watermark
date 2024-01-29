# Watermarking Detection - DNN


## Authors

- Bastien Pouëssel
- Arnaud Baradat
- Nicolas Fidel
- Tom Genlis
- Théo Ripoll
- Quentin Fisch

## Dataset creation

The characteristics of the dataset we will generate are the following:
1. Single and multi watermarked images
2. Overlapping and non-overlapping watermarks
3. Non aligned bounding boxes (4 coordinates)
4. 256x256 images
5. Base images from coco (all classes)
6. Logos from this dataset: https://hangsu0730.github.io/qmul-openlogo/

The dataset contains subdatasets for:
- Text watermarks
- Logo watermarks
- Text and logo watermarks
- Tile watermarks

We worked on multiple properties of the watermarks:
- Color
- Size
- Position
- Rotation
- Opacity
- Font
- Pattern (for tiles)

## Model

We fine-tuned a YoloV8 and DETR model on the dataset we created.
- YoloV8 -> PyTorch
- DETR -> PyTorch (https://www.kaggle.com/code/shnakazawa/object-detection-with-pytorch-and-detr#Define-Model-Components)
- DinoV2
- Retina Net -> PyTorch
- Mask R CNN -> PyTorch

