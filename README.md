# FraudSense: Advanced Financial Fraud Detection System

## Overview

FraudSense is an end-to-end machine learning system designed to detect fraudulent financial transactions in large-scale banking datasets. The project follows a complete data science workflow, including exploratory data analysis, feature engineering, model development, optimization, explainability, and robustness testing.

The system is built using the PaySim Fraud Detection Dataset and focuses on maximizing fraud detection recall while maintaining strong precision and scalability.

---

## Project Objectives

* Detect fraudulent financial transactions accurately.
* Handle severe class imbalance present in fraud datasets.
* Engineer meaningful behavioral and transactional features.
* Compare multiple machine learning algorithms.
* Optimize model performance using hyperparameter tuning.
* Evaluate robustness under real-world stress conditions.
* Build a scalable fraud detection pipeline suitable for production environments.

---

## Dataset

**Source:** PaySim Financial Transaction Dataset (Kaggle)

### Dataset Characteristics

* ~6.3 Million transactions
* Highly imbalanced fraud distribution
* Multiple transaction types:

  * CASH_IN
  * CASH_OUT
  * PAYMENT
  * TRANSFER
  * DEBIT

### Target Variable

`isFraud`

* 0 → Legitimate Transaction
* 1 → Fraudulent Transaction

---

## Project Structure

```text
FraudSense/
│
├── eda_phase1.py
├── feature_engineering.py
├── phase3_modeling.py
├── phase4_stress_testing.py
│
├── eda_outputs/
├── fe_outputs/
├── model_outputs/
├── stress_test_outputs/
│
├── LICENSE
├── .gitignore
└── README.md
```

---

## Phase 1: Exploratory Data Analysis

The first phase focuses on understanding the dataset and identifying fraud patterns.

### Analysis Performed

* Dataset inspection
* Memory optimization
* Missing value analysis
* Class imbalance analysis
* Univariate analysis
* Outlier detection
* Transaction type analysis
* Time-based fraud analysis
* Correlation analysis
* Fraud behavior exploration

### Outputs

* Statistical summaries
* Fraud distribution visualizations
* Time-pattern analysis
* Transaction-type fraud rates
* Correlation heatmaps

---

## Phase 2: Feature Engineering

Custom features were engineered to improve fraud detection performance.

### Features Created

#### Transaction Features

* Balance difference (sender)
* Balance difference (receiver)
* Amount-to-balance ratio
* Zero balance indicators

#### Error Features

* Sender balance inconsistency
* Receiver balance inconsistency
* Absolute error measurements

#### Temporal Features

* Hour of transaction
* Day of transaction
* Night transaction flag

#### Behavioral Features

* Sender transaction count
* Sender average transaction amount
* Receiver transaction statistics
* Velocity-based transaction metrics

#### Interaction Features

* High-value transaction flags
* High-risk transaction combinations
* Balance depletion interactions

#### Log-Transformed Features

Applied logarithmic transformations to highly skewed financial variables.

---

## Phase 3: Modeling & Optimization

Multiple machine learning models were trained and compared.

### Models Implemented

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

### Techniques Used

* Stratified sampling
* Class weighting
* Undersampling
* Cross-validation
* Randomized Search CV
* Grid Search Optimization
* Threshold tuning

### Evaluation Metrics

* Recall
* Precision
* F1 Score
* ROC-AUC
* PR-AUC

### Explainability

* Feature Importance Analysis
* SHAP-based model interpretation

---

## Phase 4: Stress Testing & Robustness Evaluation

The best-performing model is subjected to challenging real-world scenarios.

### Tests Conducted

#### Extreme Class Imbalance

Evaluation under:

* 0.5% fraud rate
* 0.1% fraud rate
* 0.01% fraud rate

#### Noise Injection

* Low noise
* Medium noise
* High noise
* Extreme noise

#### Concept Drift Simulation

* Changing fraud patterns
* Transaction behavior shifts
* Balance distribution changes

#### Adversarial Fraud Scenarios

Testing fraud samples specifically designed to evade detection.

#### Data Corruption & Missing Values

Assessing model resilience to incomplete information.

#### Scalability & Latency Benchmarking

Measuring:

* Prediction speed
* Memory usage
* Large-scale inference performance

---

## Technologies Used

### Programming Language

* Python 3.x

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost
* LightGBM

### Model Persistence

* Joblib

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/FraudSense.git
cd FraudSense
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib pyarrow
```

---

## Usage

### Run EDA

```bash
python eda_phase1.py
```

### Run Feature Engineering

```bash
python feature_engineering.py
```

### Train Models

```bash
python phase3_modeling.py
```

### Perform Stress Testing

```bash
python phase4_stress_testing.py
```

---

## Key Highlights

* End-to-end fraud detection pipeline
* Handles highly imbalanced datasets
* Advanced feature engineering
* Multiple ML model comparison
* Hyperparameter optimization
* Explainable AI integration
* Real-world robustness testing
* Production-oriented evaluation framework

---

## Future Improvements

* Real-time fraud detection API
* Stream processing with Apache Kafka
* Deep learning models (LSTM, Transformer)
* Online learning for concept drift adaptation
* Cloud deployment using AWS or Azure
* Dashboard integration with Streamlit

---

## License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

## Author

Developed as part of an advanced Machine Learning and Fraud Analytics project focused on building robust, scalable, and explainable fraud detection systems.
