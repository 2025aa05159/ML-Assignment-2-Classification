# ML Assignment 2 - Classification Model Deployment

**Submitted by:** Suresh Kumar
**ID:** 2025aa05159 
**EMAIL:** 2025aa05159@wilp.bits-pilani.ac.in

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
    * This dataset is built from microscopic images of breast tissue samples. It tracks 30 specific characteristics of the cell nuclei—such as their size, shape, and roughness. The goal is to use these measurements to determine if a tumor is Malignant (cancerous) or Benign (safe).
    * **Target Variable:** 'target' (0 = Malignant (Cancerous), 1 = Benign (Non-Cancerous))
    * **Number of Features:** 30 - The dataset describes 10 physical properties of the cell nuclei. For each property, 3 statistics are calculated: Mean (average), SE (Standard Error/variability), and Worst (largest/most extreme value found in the sample)
      (All numeric- radius, texture, perimeter, area, smoothness, Compactness, Concavity, Concave Points, Symmetry and Fractal Dimension)
    * **Number of Instances:** 569

## 3. Models Used & Evaluation Metrics
The following six classification models were implemented and evaluated on the test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** |  0.9737 | 0.9974 | 0.9737 | 0.9737 | 0.9736| 0.9439 |
| **Decision Tree** | 0.9474 | 0.9324 | 0.9488 | 0.9474 | 0.9468 | 0.8885 |
| **KNN** | 0.9649 | 0.9807 | 0.9649 | 0.9649 | 0.9649 | 0.9253 |
| **Naive Bayes** |0.9649 | 0.9974 | 0.9652 | 0.9649 | 0.9647 | 0.9253 |
| **Random Forest** | 0.9649 | 0.9953 | 0.9652 | 0.9649 | 0.9647 | 0.9253 |
| **XGBoost** | 0.9561 | 0.9908 | 0.9561 | 0.9561 | 0.9560 | 0.9064|

## 4. Observations & Model Comparison

### **Why we chose Recall as our main metric**
When dealing with medical data—especially cancer detection—accuracy is not the only thing that matters. hence decided to prioritize **Recall (Sensitivity)** as most important metric.

Justification: In this context, a **False Negative** means we tell a patient they are safe when they actually have cancer. That is the most dangerous mistake a model can make. A high Recall score means our model is minimizing those dangerous mistakes and catching as many positive cases as possible.

### **How the models performed**
Below are our observations on how each algorithm handled the dataset:

| ML Model Name | Analysis & Key Takeaways |
| :--- | :--- |
| **Logistic Regression** | **The Champion (Recall: 97.37%).** Surprisingly, this simple linear model outperformed the more complex ones. It gave us the best Recall and Accuracy, proving that the dataset is "linearly separable" (the classes are easy to distinguish with a straight line). |
| **Decision Tree** | **Struggled a bit (Recall: 94.74%).** This was our weakest performer. Single decision trees often "overthink" or memorize the training data (overfitting), which likely caused it to make more mistakes on the test set. |
| **KNN** | **Solid Performance (Recall: 96.49%).** It tied for second place. By setting `k=9`, the model was able to ignore noise and focus on the local patterns effectively. |
| **Naive Bayes** | **Better than expected (Recall: 96.49%).** Even though this model assumes all features are independent (which isn't always true in medical data), it performed impressively well, matching the complex ensemble models. |
| **Random Forest** | **Reliable & Stable (Recall: 96.49%).** As expected, this model improved on the single Decision Tree. By averaging 100 different trees, it smoothed out the errors and gave a much more stable prediction. |
| **XGBoost** | **Good, but not the best (Recall: 95.61%).** We often expect XGBoost to win, but in this specific case, it was slightly beaten by Logistic Regression. This shows that sometimes, a complex model is "overkill" for a smaller, cleaner dataset. |

## 5. Project Structure
This repository follows the required directory structure:

```text
BITS-ML-Assignment-2/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # List of Python dependencies
├── README.md                     # Project documentation
│
└── model/                        # Folder containing notebooks and saved binaries
    ├── <model>.ipynb             # Jupyter Notebook used for training & analysis
    ├── scaler.pkl                # Saved StandardScaler object (Critical for preprocessing)
    ├── logistic_regression.pkl   # Trained Logistic Regression model
    ├── decision_tree.pkl         # Trained Decision Tree model
    ├── knn_model.pkl             # Trained K-Nearest Neighbors model
    └── naive_bayes_model.pkl     # Trained Gaussian Naive Bayes model
    └── random_forest_model.pkl   # Trained Random Forest model
    └── xgboost_model.pkl         # Trained XGBoost model
```

## 6. How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/2025aa05159/ML-Assignment-2-Classification.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
	
## 7. ⚕️ Breast Cancer Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bits-ml-assignment-2-classification-suresh-kumar.streamlit.app/)

**Live Demo:** [Click here to launch the App 🚀](https://bits-ml-assignment-2-classification-suresh-kumar.streamlit.app/)
