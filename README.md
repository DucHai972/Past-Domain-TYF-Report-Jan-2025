# Typhoon Formation Prediction with Deep Learning

This project implements a deep learning pipeline to detect typhoons from meteorological image data. The pipeline includes data loading, preprocessing, and training a modified ResNet18 model. The goal is to classify images as either containing a typhoon (positive) or not (negative).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Running the Script Train Model (main.py)](#project-structure)
- [Running the Script Evaluate Trained Model on Full months (eval_fullyear.py)](#project-structure)

---

## Overview

This project is structured to achieve the following:

- **Data Loading and Preprocessing:**
  - Read meteorological data stored in NetCDF files using **xarray** from Path in csv datasets.
  - Convert the data into NumPy arrays for efficient handling.

- **Model Training:**
  - Adapt the ResNet18 architecture to work with meteorological data.
  - Train the model to differentiate between typhoon and non-typhoon images.

- **Performance Evaluation:**
  - Log and visualize training metrics (loss, accuracy) to assess model performance.
  - Evaluate final trained models on both segmented test data and full year predictions.


---

## Project Structure

``` bash
/N/slate/tnn3/HaiND/01-06_report/01-06_report/ # Root directory of the project 
    │── pycache/ # Compiled Python files (auto-generated) 
    │── .gitignore # Specifies files to be ignored by Git 
    │── csv/ # Folder for storing CSV files (e.g., processed data) 
    │── data/ # Raw and processed dataset storage 
    │── models/ # Contains model-related scripts and saved models 
    │── plotting/ # Scripts for visualizing results and data 
    │── result/ # Stores results from model training and evaluation 
    │── result_earlystopping/ # Stores results with early stopping applied 
    │── sandbox/ # Experimental scripts and exploratory work 
    │── srun/ # Slurm job submission scripts (for cluster runs) 
    │── utils/ # Helper functions and utility scripts

    │── config.py # Configuration settings for the project
    │── main.py # Main script to run the project  

    # --- Evaluate trained model on full year data 
    │── eval_fullyear.py # Main script to evaluate full year data
    │── result_fullmap # Result of evaluate full year data
      │── all_months # Test on data full 12 months
      │── storm_months # Test on data from May (5) to November (11)
```

## Running the Script

To train and evaluate the model, use the following command:

```bash
python main.py --time t2_rus20_cw2 --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 20 --class_weight 2
```

### Command-Line Arguments

| Argument          | Type    | Default  | Description |
|------------------|--------|---------|-------------|
| `--time`        | `str`  | Required | Identifier for the experiment (e.g., different preprocessing settings). |
| `--norm_type`   | `str`  | Required | Type of normalization (`new` or `old`). |
| `--lr`          | `float`| `1e-7`   | Learning rate for training the model. |
| `--pos_ind`     | `int`  | `1`      | Positive sample index (e.g., how early a sample is considered positive). |
| `--under_sample`| `flag` | `False`  | Enables data undersampling to balance classes. |
| `--rus`         | `int`  | `None`   | Undersampling ratio (used when `--under_sample` is enabled). |
| `--class_weight`| `int`  | `None`   | Class weight for handling imbalanced data. |
| `--small_set`   | `flag` | `False`  | If enabled, uses a smaller dataset for quick testing. |
| `--model`       | `str`  | `resnet` | Specifies the model type (default: ResNet). |

### Example Usage

To train with undersampling and specific class weights:
This is using normalised method, learning rate 1e-7, positive sample is time step t-2, using undersampling method - with ratio 1:10 and class weight balanced

```bash
python main.py --time experiment_1 --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 10 --class_weight 1
```

For a quick test using a small dataset:
This is using normalised method, learning rate 1e-7, positive sample is time step t-2, not using undersampling method, only use small subset of data and class weight balanced
```bash
python main.py --time test_run --norm_type new --lr 1e-7 --pos_ind 2 --small_set --class_weight 1
```

---

## Running the Script Evaluate Trained Model on Full months (eval_fullyear.py)
Besides the standard training and testing routine, this project includes a dedicated script for full year predictions. The file eval_fullyear.py is used as:

```bash
python eval_fullyear.py --timestep t2_rus4_cw1 --strict --fullmonth
```

### Command-Line Arguments

| Argument          | Type    | Default  | Description |
|------------------|--------|---------|-------------|
| `--timestep`        | `str`  | Required | The name of model to load (e.g., t2_rus4_cw1). |
| `--strict`   | `flag`  | None | If enabled, labels positive samples only at t-i; otherwise, t-0 to t-i. |
| `--fullmonth`          | `flag`| None   | If enabled, tests on all 12 months; otherwise, tests May to November only. |


### Output 
Results for --fullmonth enabled: ./result_fullmap/all_months
Results for --fullmonth disabled: ./result_fullmap/storm_months