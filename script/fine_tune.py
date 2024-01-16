import torch
from torchvision import models, transforms
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


def load_model():
    # Load model
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

    return model

def load_preprocess():
     # Define the standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),             # Resize to 256x256
        transforms.CenterCrop(224),         # Crop to 224x224
        transforms.ToTensor(),              # Convert to tensor
        normalize                           # Normalize pixel values
    ])

    return preprocess

def load_data():
    preprocess = load_preprocess()

    dataset = load_dataset("bastienp/visible-watermark-pita")

    # Preprocess the dataset
    dataset = dataset.map(lambda x: {'image': preprocess(x['image']), 'label': x['label']})

    # Split the dataset
    dataset.set_format('torch', columns=['image', 'label'])

    return dataset


if __name__ == "__main__":
    model = load_model()
    data = load_data()

    # Fine tune model
    training_args = TrainingArguments(
        output_dir='./fine-tuning-results',          
        num_train_epochs=3,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=data["train"],         
        eval_dataset=data["validation"]      
    )

    trainer.train()

    # Save model 
    trainer.save_model("./models/fine-tuned-detr-resnet50")

