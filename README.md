# Plant-Disease-Detection-Corn_plants
Certainly! Here's the updated README.md file with the information about using only the corn images due to resource constraints:

```
# Plant Disease Detection using VGG16 with Spatial Attention

This repository contains code for training a deep learning model for plant disease detection using the VGG16 architecture with spatial attention mechanism. The dataset used for training is a subset of the PlantVillage dataset, consisting of images of corn leaves, both diseased and healthy.

## Getting Started

To use this code, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the corn images subset from the PlantVillage dataset from [Kaggle](https://www.kaggle.com/abdallahalidev/plantvillage-dataset) and unzip it.

4. Upload the `kaggle.json` file to your Google Colab environment (if running on Colab) to access the dataset directly from Kaggle.

5. Run the provided notebook `Copy_of_Plant_Disease_Detection_(1).ipynb` to train the model.

6. After training, the model weights will be saved as `VGG_Attention_PV.h5`.

## Dataset Preparation

The dataset consists of images of corn leaves, categorized into different classes representing various diseases and healthy states. Due to resource constraints, only the corn images subset of the PlantVillage dataset is used for training. This subset contains images of diseased and healthy corn leaves.

## Model Architecture

The model architecture is based on the VGG16 convolutional neural network with an added spatial attention mechanism. This attention mechanism allows the model to focus on relevant regions of the input images, improving its performance in detecting diseases in corn leaves.

## Training and Evaluation

The model is trained using an image data generator with data augmentation techniques to increase the diversity of training data. After training, the model is evaluated on a subset of the corn images dataset to assess its performance in disease detection. The evaluation metrics include loss and accuracy.

## Results

The trained model achieves a certain level of accuracy in detecting diseases in corn leaves, which can be further improved with fine-tuning and optimization techniques.

## Acknowledgments

This project is based on the work by Abdallah Ali on the PlantVillage dataset, available on Kaggle.
```

This README emphasizes that only a subset of the PlantVillage dataset consisting of corn images is used for training due to resource constraints. Feel free to customize it further to include any additional information or instructions as needed.
