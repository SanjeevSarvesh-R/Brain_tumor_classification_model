# Brain Tumor Classification using MONAI and DenseNet121

## About the project

This project is a deep learning based brain MRI classification system. It takes a brain MRI image as input and predicts one of four classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project was developed as a college-level machine learning project to understand how medical images can be processed and classified using deep learning.

The model is built using **MONAI**, a PyTorch-based framework designed for healthcare and medical imaging applications. For image classification, the project uses **DenseNet121**.

## Technologies used

- Python
- PyTorch
- MONAI
- DenseNet121
- NumPy
- Pillow
- Scikit-learn

## Project structure

```text
Brain-Tumor-Classification/
│
├── train.py
├── model.py
├── predict.py
├── evaluate.py
├── best_brain_tumor_monai.pth
├── requirements.txt
├── README.md
├── .gitignore
│
└── scripts/
    ├── new.py
    └── test1.py
```

The training and testing image datasets are not included in this repository because of their size and dataset distribution considerations.

## Dataset

The dataset contains four classes:

1. Glioma
2. Meningioma
3. Pituitary
4. No tumor

The model was trained using images from the training dataset and evaluated separately on a test dataset.

The test set contains **1,600 images**, with 400 images from each class.

## Model

The project uses **DenseNet121** with:

- 2D image input
- 1 input channel
- 4 output classes
- Image size of 224 × 224 pixels

The MRI images are converted to grayscale before being given to the model.

During training, image augmentation such as flipping, rotation, and zooming was used to improve the model's ability to generalize to unseen images.

## Training

The model was trained for 10 epochs using:

- Batch size: 32
- Learning rate: 0.0001
- Optimizer: Adam
- Loss function: Cross Entropy Loss
- Random state: 42

The best-performing model during validation was saved as:

```text
best_brain_tumor_monai.pth
```

## Results

The trained model was evaluated on 1,600 previously unseen test images.

### Overall accuracy

**87.38%**

### Per-class accuracy

| Class | Accuracy |
|---|---:|
| Glioma | 67.50% |
| Meningioma | 83.25% |
| Pituitary | 99.00% |
| No tumor | 99.75% |

The model performed particularly well on pituitary tumor and no-tumor images. Glioma was the most difficult class for the model, with several glioma images being classified as meningioma, pituitary, or no tumor.

## How to run the project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Brain-Tumor-Classification
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Predict a single MRI image

Run:

```bash
python predict.py
```

The script allows you to select an MRI image and displays the predicted class, confidence, and probabilities for all four classes.

### 4. Evaluate the model

Make sure the testing dataset is available in the expected folder structure:

```text
Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Then run:

```bash
python evaluate.py
```

The evaluation script reports:

- Overall accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class accuracy

### 5. Train the model again

If the training dataset is available, run:

```bash
python train.py
```

The trained model will be saved as:

```text
best_brain_tumor_monai.pth
```

## Important note

This project is intended for **educational and research purposes**. The predictions are produced by a machine learning model and should not be treated as a medical diagnosis or used as a replacement for evaluation by a qualified medical professional.

The reported accuracy is based on this particular dataset and test set. Performance on other datasets or real clinical MRI scans may be different.

## Future improvements

Some possible improvements for the project include:

- Training with more diverse MRI datasets
- Experimenting with other CNN architectures
- Hyperparameter tuning
- Improving MRI preprocessing
- Adding model explainability such as Grad-CAM
- Building a web interface for image upload and prediction
- Adding proper model versioning and experiment tracking

## Project status

The machine learning pipeline is complete, including training, model saving, single-image prediction, and evaluation. The trained model can be integrated into a web application through the reusable `model.py` inference functions.
