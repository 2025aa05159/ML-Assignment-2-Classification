# ML Assignment 2 - Classification Model Deployment

**Submitted by:** [Suresh Kumar]
**ID:** [2025aa05159]

## 1. Problem Statement
The goal of this assignment is to build a complete Machine Learning pipeline for a classification problem. This involves:
1.  Selecting a real-world dataset.
2.  Implementing and training six different classification models.
3.  Evaluating models using metrics like Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4.  Deploying the best performing solution as an interactive web application using Streamlit.

## 2. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** Scikit-learn Library (Originally from UCI Machine Learning Repository)
* **Link:** [Scikit-learn Documentation - load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
* **Description:**
    * This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. It describes the characteristics of the cell nuclei present in the image to predict if a tumor is malignant or benign.
    * **Target Variable:** 'target' (0 = Malignant (Cancerous), 1 = Benign (Non-Cancerous))
    * **Number of Features:** 30 (All numeric, e.g., radius, texture, perimeter, area, smoothness, etc.)
    * **Number of Instances:** 569

## 3. Models Used & Evaluation Metrics
The following six classification models were implemented and evaluated on the test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Decision Tree** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **KNN** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Naive Bayes** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Random Forest** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **XGBoost** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

*(Note: Replace '0.00' with your actual calculated metrics from Phase 2)*

## 4. Observations
Observations on the performance of each model based on the metrics above.

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | [e.g., Provided a baseline accuracy but struggled with non-linear relationships.] |
| **Decision Tree** | [e.g., Showed signs of overfitting with high training accuracy but lower test accuracy.] |
| **KNN** | [e.g., Performance varied significantly based on the value of 'k'.] |
| **Naive Bayes** | [e.g., Performed surprisingly well/poorly given the assumption of feature independence.] |
| **Random Forest** | [e.g., Improved stability and accuracy over the single Decision Tree.] |
| **XGBoost** | [e.g., Achieved the highest AUC score, handling class imbalance effectively.] |

## 5. Project Structure
This repository follows the required directory structure:
project-folder/ 
  ├── app.py # Main Streamlit application 
  ├── requirements.txt # List of dependencies 
  ├── README.md # Project documentation 
  └── model/ # Folder containing saved .pkl models 
    ├── logistic_regression.pkl 
    ├── decision_tree.pkl 
    ├── ...


## 6. How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [Your Repo Link]
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
