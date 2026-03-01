<p align="center">
  <img src="https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/heart-disease.jpeg" alt="k26-heart-disease-classification" width="600"/>
</p>

<br />
<p align="center">
  <h3 align="center">HeArT dIsEaSe</h3>
  <p align="center">Logistic Regression Heart Disease Classifier</p>
</p>

## Introduction

**k26-heart-disease-classification** is a machine learning project built for the Kaggle Playground Series competition:
**Predicting Heart Disease (Season 6, Episode 2)**.

The objective of this project is to predict the probability of heart disease using a classical logistic regression model.

This project demonstrates a complete end-to-end machine learning workflow for tabular classification, including:

- data preprocessing
- feature engineering
- model training
- ROC-AUC evaluation
- prediction file generation for Kaggle submission

The model is evaluated using **Area Under the ROC Curve (ROC-AUC)**, as specified in the competition guidelines.

This project focuses on building a strong foundational understanding of tabular machine learning using interpretable statistical models rather than deep learning architectures.

---

## Getting Started


## Project Structure

- [`README.md`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/README.md) – Main documentation and usage instructions.
- [`pyproject.toml`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/pyproject.toml) – Project configuration and dependency definitions.
- [`train.py`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/train.py) – Main training script for the logistic regression model | Local
- [`train.sh`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/train.sh) – Main training script for cluster execution | SLURM
- [`model/`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/tree/main/model) – Model definitions.
  - [`heart_disease.py`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/model/heart_disease.py) – Core logistic regression implementation.
- [`utils/`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/tree/main/utils) – Helper modules for preprocessing and evaluation.
  - [`lr_utils.py`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/utils/lr_utils.py) – Logistic regression utilities.
  - [`visualize.py`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/blob/main/utils/visualize.py) – Visualization utilities for evaluation metrics.
- [`outputs/`](https://github.com/MosesTheRedSea/k26-heart-disease-classification/tree/main/outputs) – Training logs, results, and submission outputs.

---

## Installation

Clone the repo

```sh
git clone https://github.com/MosesTheRedSea/k26-heart-disease-classification.git
