# Satellite Image Classification with CNNs

Final project for **Programming in Python II** at **Johannes Kepler University Linz**.

## Overview

This project implements an end-to-end computer vision pipeline for satellite image classification using **Python** and **PyTorch**.

The model classifies 64×64 RGB satellite images into 10 land-use categories:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Project Pipeline

The project includes:

- Data preprocessing and stratified train/validation splitting
- Exploratory Data Analysis (EDA)
- Custom CNN implementation in PyTorch
- Training and validation workflows
- Model evaluation and error analysis
- Confusion matrix and misclassified-sample analysis
- Test-set inference
- Interactive Shiny web application for image upload and prediction

## Results

The model achieved approximately **93.7% validation accuracy** across the 10 land-use classes.

## Technologies

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- Matplotlib
- PIL
- Shiny for Python


## Repository Structure

```text
app/
assets/
satellite_image_classification.ipynb
```

## Academic Context

This project was developed as the final project for the course **Programming in Python II**
at **Johannes Kepler University Linz**.
