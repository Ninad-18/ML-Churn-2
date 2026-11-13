## 1. Project Overview

This mini project focuses on predicting customer churn in the telecommunications industry using Data Science and Machine Learning techniques.

Customer churn refers to customers discontinuing a company's services. Predicting churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

The project uses the **Telco Customer Churn Dataset** and implements multiple classification algorithms, including traditional Machine Learning models and an Artificial Neural Network (ANN).

## 2. Objectives

- Understand and analyze the Telco Customer Churn dataset.
- Perform data cleaning and preprocessing.
- Handle missing values, duplicate records, and outliers.
- Perform Exploratory Data Analysis (EDA) using visualizations.
- Handle class imbalance using SMOTE.
- Divide the dataset into training, validation, and testing sets.
- Implement and compare multiple classification algorithms.
- Evaluate model performance using standard classification metrics.
- Identify the most suitable model for churn prediction.

## 3. Dataset Description

**Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information related to:

| Feature Category     | Examples                                                            |
|----------------------|---------------------------------------------------------------------|
| Customer Information | Customer ID, gender, senior citizen, partner, dependents            |
| Account Information  | Tenure, contract, payment method, paperless billing                 |
| Services             | Phone service, internet service, streaming services, online service |                                                                    
| Charges              | Monthly charges, total charges                                      |
| Target Variable      | Churn                                                               |

The target variable is `Churn`:
- `Yes` – Customer has churned.
- `No` – Customer has not churned.

## 4. Technologies and Libraries Used

- **Python** – Programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical operations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical visualization.
- **Scikit-learn** – Machine Learning algorithms and preprocessing.
- **Imbalanced-learn** – SMOTE for handling class imbalance.
- **TensorFlow / Keras** – Artificial Neural Network implementation.

## 5. Data Preprocessing

The following preprocessing operations are performed:

1. Load and inspect the dataset.
2. Remove duplicate records.
3. Remove the `customerID` column because it is an identifier.
4. Convert `TotalCharges` into numeric format.
5. Handle missing values using median and most-frequent-value imputation.
6. Detect and cap outliers using the Interquartile Range (IQR) method.
7. Encode categorical variables using One-Hot Encoding.
8. Standardize numerical features using StandardScaler.
9. Convert the target variable into binary values.

## 6. Exploratory Data Analysis

EDA is performed to understand customer behavior and identify patterns related to churn.

Visualizations include:

- Customer churn distribution.
- Customer tenure distribution.
- Monthly charges versus churn.
- Total charges versus churn.
- Contract type versus churn.
- Internet service versus churn.
- Correlation heatmap.

## 7. Handling Class Imbalance

The dataset contains more non-churning customers than churning customers.

**SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training dataset to generate synthetic minority-class observations and improve the model's ability to identify churn cases.

SMOTE is not applied to validation or testing data.

## 8. Dataset Partitioning

The dataset is divided into three subsets:

| Dataset    | Percentage |
|------------|------------|
| Training   | 60%        |
| Validation | 20%        |
| Testing    | 20%        |

- Training data is used to train the models.
- Validation data is used for model selection.
- Testing data is used for final performance evaluation.

## 9. Machine Learning Models Implemented

The following classification algorithms are implemented:

1. **Logistic Regression** – A linear classification algorithm used as a baseline.
2. **Decision Tree** – Uses decision rules to classify customers.
3. **Random Forest** – Combines multiple decision trees to improve predictive performance.
4. **AdaBoost** – Uses sequential weak learners to improve classification.
5. **K-Nearest Neighbors (KNN)** – Classifies customers based on their nearest neighbors.
6. **Artificial Neural Network (ANN)** – Uses interconnected neurons and hidden layers to learn complex patterns.

### ANN Architecture

The ANN consists of:

- Input Layer – Accepts preprocessed customer features.
- Hidden Layer 1 – 64 neurons with ReLU activation.
- Dropout Layer – 30% dropout.
- Hidden Layer 2 – 32 neurons with ReLU activation.
- Dropout Layer – 20% dropout.
- Output Layer – 1 neuron with Sigmoid activation.

The ANN uses the Adam optimizer and Binary Cross-Entropy loss function.

## 10. Model Evaluation Metrics

The models are evaluated using:

| Metric           |  Purpose                                                           |
|------------------|--------------------------------------------------------------------|
| Accuracy         | Measures overall correct predictions.                              |
| Precision        | Measures how many predicted churn cases are actually churn cases.  |
| Recall           | Measures how many actual churn cases are correctly identified.     |
| F1 Score         | Balances precision and recall.                                     |
| ROC-AUC          | Measures the model's ability to distinguish between churn classes. |
| Confusion Matrix | Displays correct and incorrect classifications.                    |

Model performance is compared using tables and visualizations.

## 11. Expected Project Output

The project produces:

- Dataset information and statistical summaries.
- Missing-value and duplicate analysis.
- EDA visualizations.
- Class distribution before and after SMOTE.
- Training, validation, and testing results.
- Classification reports for the implemented models.
- Confusion matrices.
- ROC curves.
- Model performance comparison charts.
- Final model selection based on evaluation metrics.

## 12. Conclusion

This mini project demonstrates how Data Science and Machine Learning can be applied to customer churn prediction.

Multiple classification algorithms and an Artificial Neural Network are implemented and evaluated using accuracy, precision, recall, F1 score, and ROC-AUC.

The most suitable model is selected based on its actual test performance, with particular importance given to F1 score and recall because correctly identifying customers likely to churn is important for customer retention.

The project demonstrates the complete Data Science workflow, from raw dataset exploration and preprocessing to model training, evaluation, and comparison.

## 13. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

### Step 2: Place the Dataset

Keep the following CSV file in the same directory as your notebook or Python script:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Step 3: Run the Code

Execute all notebook cells in sequence, or run the Python script.

### Step 4: Analyze Results

Review the model comparison table, evaluation metrics, confusion matrices, and visualizations to identify the best-performing model.

---

**Author:** Ninad Patil
