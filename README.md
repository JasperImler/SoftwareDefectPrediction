# Software Defect Prediction Model

This project implements a machine learning-based software defect prediction model using NASA datasets. The model is designed to identify potential defects in software modules based on various software metrics.

## Background

Software defect prediction is a crucial area in software quality assurance that aims to identify defect-prone modules before they are released to users. By accurately predicting which modules are likely to contain defects, development teams can allocate testing resources more efficiently and improve software quality. This project was developed as a final assignment for a Software Testing course.

## Dataset

The project uses NASA defect datasets, which contain software metrics and defect information from various NASA projects. These datasets include metrics such as:
- Lines of code
- McCabe complexity measures
- Halstead complexity measures
- Code churn metrics
- And defect labels (defective or non-defective)

## Project Structure

```
├── arff_to_csv_converter.py    # Tool to convert ARFF files to CSV format
├── data_loader.py              # Utility for loading and preprocessing data
├── data_splitter.py            # Tool for splitting data into train/test sets
├── data_standardizer.py        # Data standardization tool
├── model_tuner.py              # Model parameter tuning with cross-validation
├── model_tuner_interactive.py  # Interactive interface for model tuning
├── OriginalData/               # Original NASA datasets in ARFF format
├── CleanedData/                # Cleaned datasets
├── CSV/                        # Datasets converted to CSV format
├── Split/                      # Train-validation-test split datasets
├── Standardized/               # Standardized datasets
├── Models/                     # Trained models and evaluation results
└── README.md
```

## Workflow

The complete workflow for this project consists of the following steps:

1. **Data Conversion** (if needed):
   ```bash
   python arff_to_csv_converter.py
   ```
   Converts the original ARFF files to CSV format for easier processing.

2. **Data Loading and Cleaning**:
   ```bash
   python data_loader.py
   ```
   Loads the datasets, handles missing values, and performs basic preprocessing.

3. **Train-Test Splitting**:
   ```bash
   python data_splitter.py --dataset CM1
   ```
   Splits a specific dataset into training, validation, and test sets.

4. **Data Standardization**:
   ```bash
   python data_standardizer.py --dataset CM1 --verify
   ```
   Standardizes the dataset and verifies the standardization results.

5. **Model Tuning**:
   ```bash
   python model_tuner.py --input Standardized/CM1/standardized_data.npz --model rf
   ```
   Tunes a Random Forest model using grid search and cross-validation.

6. **Interactive Model Tuning** (alternative to step 5):
   ```bash
   python model_tuner_interactive.py
   ```
   Launches an interactive interface for dataset selection and model tuning.

## Features

1. **Data Processing Pipeline**:
   - Converting ARFF to CSV format
   - Loading and cleaning NASA defect datasets
   - Handling missing values
   - Train-validation-test splitting
   - Feature standardization (zero mean and unit variance)
   - Verification of standardization results

2. **Model Parameter Tuning**:
   - Hyperparameter optimization using grid search and cross-validation
   - Support for multiple machine learning algorithms:
     - Random Forest
     - Support Vector Machine
     - Logistic Regression
     - K-Nearest Neighbors
     - Decision Tree
     - Gradient Boosting
   - Progress tracking with progress bars

3. **Model Evaluation**:
   - Comprehensive performance metrics:
     - Accuracy
     - Precision
     - Recall
     - F1 score
     - AUC-ROC
     - Matthews Correlation Coefficient
   - Visualization of results:
     - Confusion matrix
     - ROC curve
     - Performance metrics comparison
     - Parameter search results

4. **User Interface**:
   - Interactive command-line interface for model selection and tuning
   - Batch processing capability for multiple datasets
   - Customizable evaluation criteria

## Advanced Options

- **Quick Mode**:
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model rf --reduce_params
  ```
  Reduces parameter combinations to speed up training while maintaining coverage.

- **Random Search**:
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model svm --random --n_iter 20
  ```
  Uses random search instead of grid search with 20 iterations.

- **Custom Scoring**:
  ```bash
  python model_tuner.py --input Standardized/CM1/standardized_data.npz --model lr --scoring roc_auc
  ```
  Uses AUC-ROC as the scoring metric instead of the default F1 score.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn (optional)
- tqdm

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib tqdm seaborn
```

## Results

The model generates comprehensive evaluation reports including:
- Classification report with precision, recall, and F1 score for each class
- Confusion matrix visualization
- ROC curve for models that support probability prediction
- Performance metrics comparison chart

Results are saved in the `Models/{dataset}/{model}/` directory.

## Future Work

- Implementation of ensemble methods combining multiple models
- Deep learning approaches for defect prediction
- Feature importance analysis for better interpretability
- Time-based evaluation for software evolution studies

## License

This project is provided for educational purposes and is available under the MIT License.

## Acknowledgments

- NASA for providing the defect datasets
- The scikit-learn team for their excellent machine learning library
- All contributors to this project 
