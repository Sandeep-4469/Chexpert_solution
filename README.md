# Automated Lung Abnormality Detection Using Deep Learning

## Project Overview
This project focuses on developing an automated system for detecting lung abnormalities in chest X-ray images using deep learning. The model leverages a **DenseNet121** architecture that has been fine-tuned for multi-label classification. The goal is to detect six specific lung diseases from X-ray images:

- **Atelectasis**
- **Cardiomegaly**
- **Consolidation**
- **Edema**
- **Pleural Effusion**
- **Pneumonia**

This system is built using the **CheXpert** dataset, which is publicly available and contains a large collection of labeled chest X-ray images. The final goal is to automate the process of detecting these diseases in clinical settings, improving diagnostic efficiency and accuracy.

## Dataset

The **CheXpert** dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert). This dataset contains over 200,000 chest X-ray images, each annotated with labels for a range of lung diseases. In this project, we focus on the following six diseases:

- **Atelectasis**
- **Cardiomegaly**
- **Consolidation**
- **Edema**
- **Pleural Effusion**
- **Pneumonia**

### Dataset Structure

The dataset is organized into training and validation directories, with the following structure:

Dataset/\
├── train/\
│   ├── Atelectasis/\
│   ├── Cardiomegaly/\
│   ├── Consolidation/\
│   ├── Edema/\
│   ├── Pleural_Effusion/\
│   ├── Pneumonia/\
│   ├── healthy/\
│   \
├── valid/\
│   ├── Atelectasis/\
│   ├── Cardiomegaly/\
│   ├── Consolidation/\
│   ├── Edema/\
│   ├── Pleural_Effusion/\
│   ├── Pneumonia/\
│   ├── healthy/\
│\
|── train.csv\
|── valid.csv\

Further change in this dataset \

Dataset\
├── Atelectasis\
│   ├── disease\
│   └── no_finding\
├── Cardiomegaly\
│   ├── disease\
│   └── no_finding\
├── Consolidation\
│   ├── disease\
│   └── no_finding\
├── Edema\
│   ├── disease\
│   └── no_finding\
├── Pleural_Effusion\
│   ├── disease\
│   └── no_finding\
├── Pneumonia\
│   ├── disease\
│   └── no_finding\
└── train.csv\
└── valid.csv\








### Dataset Link:
- [CheXpert Dataset on Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)

## Methodology

### 1. Data Preprocessing
To enhance the model’s generalization ability, data augmentation techniques were used:
- **Random rotation** of images (up to 20 degrees)
- **Random resized crop** to 224x224 pixels
- **Random horizontal flip**
- **Normalization** based on ImageNet's mean and standard deviation values.

### 2. Model Architecture
- **Base Model**: **DenseNet121** pre-trained on ImageNet was used as the backbone model for feature extraction.
- **Fine-tuning**: The final classifier layers were replaced with a custom multi-label classification head.
  - The output layer used a **Sigmoid** activation to handle multi-label classification.
  - The architecture used a **dropout layer** for regularization and to prevent overfitting.

### 3. Training Strategy
- The model was trained using the **Adam optimizer** with a learning rate of `1e-4`.
- **Loss Function**: Binary Cross-Entropy loss, as this is a multi-label classification task.
- **Metrics**: Accuracy, Precision, Recall, and F1 Score were used to evaluate model performance.

### 4. Model Evaluation
The trained model was evaluated on the validation set using the following metrics for each disease:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics provide insights into the model's ability to detect each disease individually, as well as its overall performance.

### 5. Multi-label Classification
Since this is a multi-label classification problem, the model was designed to predict the probability of each disease independently. Each disease is treated as a separate binary classification task, and the final predictions are obtained through the Sigmoid function, which outputs values between 0 and 1.

## Results

The performance of the model on the validation set is summarized in the table below:

| Disease         | Accuracy | Precision | Recall  | F1 Score |
|-----------------|----------|-----------|---------|----------|
| Atelectasis     | 0.7966   | 0.6346    | 0.8684  | 0.7333   |
| Cardiomegaly    | 0.8113   | 0.6800    | 0.8947  | 0.7727   |
| Consolidation   | 0.8732   | 0.8085    | 1.0000  | 0.8941   |
| Edema           | 0.9277   | 0.8636    | 1.0000  | 0.9268   |
| Pleural Effusion| 0.8286   | 0.7381    | 0.8158  | 0.7750   |
| Pneumonia       | 0.9565   | 0.9500    | 1.0000  | 0.9744   |

### Key Observations:
- The model achieved the highest accuracy for **Pneumonia** (95.65%) and the highest F1 score for **Pneumonia** (97.44%).
- **Consolidation** and **Edema** achieved perfect recall, indicating that the model correctly identified all instances of these diseases.
- Overall, the model demonstrated strong performance across all six diseases, with especially high precision and recall for **Pneumonia** and **Edema**.

## Instructions

### Training the Model
To train the model, follow these steps:
1. Prepare the dataset and ensure the file paths are correctly set in the training script.
2. Run the training script using your preferred environment (GPU recommended).

### Evaluating the Model
Once the model is trained, evaluate its performance on the validation set. The evaluation script will print out the accuracy, precision, recall, and F1 score for each disease.

### Model Inference
After training, the model can be used to predict diseases in new chest X-ray images. The model outputs probabilities for each disease, where values close to 1 indicate the presence of the disease.

## Conclusion

This project demonstrates the application of deep learning for detecting lung abnormalities from chest X-ray images. The model performs well in multi-label classification tasks, achieving high accuracy, precision, recall, and F1 scores across multiple diseases. The results show that the model is effective for clinical applications, particularly in the detection of **Pneumonia** and **Edema**, where it achieved near-perfect recall.

In future work, the model could be improved by exploring more advanced architectures or incorporating additional datasets for greater generalization. The model could also be integrated into clinical systems to assist healthcare professionals in diagnosing lung diseases more efficiently.


## Acknowledgements

- The **CheXpert** dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert).
- Special thanks to the creators and contributors of the **CheXpert** dataset for making it publicly available.
