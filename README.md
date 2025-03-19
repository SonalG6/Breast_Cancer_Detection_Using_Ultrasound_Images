# Breast Cancer Detection Using Ultrasound Images

## Project Overview
This project aims to develop a robust deep learning model to detect breast cancer using ultrasound images. Leveraging a dataset sourced from Kaggle, the model classifies ultrasound images into three categories:

- **Normal**
- **Benign**
- **Malignant**

The project demonstrates the potential of AI in medical imaging, offering a non-invasive and efficient approach to early breast cancer detection.

## Dataset
The dataset used in this project is publicly available on Kaggle: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data).

### Dataset Details:
- Contains 780 grayscale ultrasound images.
- Divided into three categories: **Normal**, **Benign**, and **Malignant**.
- Includes segmentation masks for precise region-of-interest detection.

### Preprocessing:
- Images were resized to 128x128 pixels for uniformity.
- Normalization was applied to scale pixel values between 0 and 1.
- Data augmentation techniques (e.g., horizontal/vertical flipping, rotation, and zooming) were applied to enhance model generalization and prevent overfitting.

## Model Architecture
A Convolutional Neural Network (CNN) was designed and trained to classify the ultrasound images. The architecture includes:

- **Convolutional Layers**: For feature extraction with filters of increasing complexity.
- **Pooling Layers**: For dimensionality reduction and retaining important features.
- **Dropout Layers**: To prevent overfitting by randomly disabling neurons during training.
- **Fully Connected Layers**: For final classification into three categories.
- **Activation Function**: ReLU for hidden layers and softmax for the output layer.
- **Optimizer**: Adam for adaptive learning rate optimization.
- **Loss Function**: Categorical cross-entropy to handle multi-class classification.

## Training Details
- **Epochs**: 200
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Framework**: TensorFlow/Keras

### Key Observation:
- The model’s accuracy consistently improved as the number of epochs increased. After 200 epochs, the model achieved an impressive accuracy of **98%** on the test set.
- This demonstrates the importance of sufficient training time for deep learning models to converge effectively.

## Results
The model was evaluated on a separate test set, achieving:
- **Accuracy**: 98%
- **Precision**: High for both benign and malignant classes, ensuring minimal false positives.
- **Recall**: High, indicating minimal false negatives, which is critical in medical applications.

### Confusion Matrix
The confusion matrix provided insights into class-wise performance, showing:
- Excellent differentiation between benign and malignant cases.
- Minimal misclassifications for the “Normal” class.

### Performance Metrics
- **F1 Score**: Balanced performance across precision and recall.
- **ROC-AUC**: Demonstrated strong discriminatory ability between classes.

## How to Use
### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (for image preprocessing)

### Installation
Clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
