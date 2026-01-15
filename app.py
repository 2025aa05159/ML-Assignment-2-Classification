import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# 1. ADDED: matthews_corrcoef and roc_auc_score imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score

# 1. Page Configuration
st.set_page_config(page_title="BITS ML Assignment-2 | Breast Cancer Predictor", page_icon="⚕️", layout="wide")

st.title("Machine Learning |Classification Model Evaluation - Breast Cancer Predictor")
st.markdown("Upload a CSV test dataset to evaluate the performance of the trained models.")

# --- Page SIDEBAR: Configuration ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/330px-BITS_Pilani-Logo.svg.png", use_container_width=True)
st.sidebar.header("Configuration")

# Option to use built-in data
use_demo_data = st.sidebar.checkbox("Using Demo Data (Built-in)", value=False)
# Dataset Upload Option
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV) for Breast Cancer Prediction", type=["csv"])

# Model Selection Dropdown
model_options = ["Logistic Regression", "Decision Tree","KNN Model", "Naive Bayes", "Random Forest","XGBoost"]
selected_model_name = st.sidebar.selectbox("Select Model", model_options)

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---") # Adds a visual separator line
st.sidebar.markdown("### 📧 Contact Developer")
st.sidebar.markdown(
    """
    **Student Name** : Suresh Kumar 
    [2025aa05159@wilp.bits-pilani.ac.in](mailto:2025aa05159@wilp.bits-pilani.ac.in)
    """
)

# Helper function: Load Model & Scaler
@st.cache_resource
def load_resources(model_name):
    scaler = None
    model = None
    
    if model_name == "Logistic Regression":
        model_path = "model/logistic_regression.pkl"
    elif model_name == "Decision Tree":
        model_path = "model/decision_tree.pkl"
    elif model_name == "KNN Model":
        model_path = "model/knn-model.pkl"
    elif model_name == "Naive Bayes":
        model_path = "model/naive_bayes_model.pkl"
    elif model_name == "Random Forest":
        model_path = "model/random_forest_model.pkl"
    elif model_name == "XGBoost":
        model_path = "model/xgboost_model.pkl"
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Models binary files: {e}")
        return None, None

    return model, scaler

# --- MAIN LOGIC ---

if uploaded_file is not None or use_demo_data:
    
    # --- STEP 1: LOAD DATA ---
    if use_demo_data:
        # Load data directly from Sklearn
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        
        data = load_breast_cancer()
        X_full = pd.DataFrame(data.data, columns=data.feature_names)
        y_full = pd.Series(data.target)

        # Ignore the training part data and only keep the test data
        _, X_test_demo, _, y_test_demo = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

          # Reconstruct the DataFrame for the App to use this test data
        df = X_test_demo.copy()
        df['target'] = y_test_demo  # 0=Malignant, 1=Benign
        st.info("✅ Using ONLY the unseen Test Split (20%) of the built-in dataset.")
   
        
        # Display preview
        st.write("### Preview of Demo Data")
        st.dataframe(df.head())
        
        # Define target column automatically for demo data
        target_col = 'target'
        
    else:
        # Load user provided csv data
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ CSV file uploaded successfully.")
        
        # Preview of uploaded csv file
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())
        
        # Let user select target column manually
        target_col = st.sidebar.selectbox("Select Target Column (Y)", df.columns, index=len(df.columns)-1)

    # --- STEP 2: COMMON LOGIC (Runs for BOTH Demo and Upload CSV) ---
    
    # Separate Features (X) and Target (y)
    y_true = df[target_col]
    X = df.drop(columns=[target_col])

    # Load Resources
    model, scaler = load_resources(selected_model_name)

    if model and scaler:
        # Preprocessing (Scaling)
        try:
            X_scaled = scaler.transform(X)
            
            # Make Predictions (Labels)
            y_pred = model.predict(X_scaled)

            # Make Probability Predictions (Needed for AUC Score)
            # We grab the probability for class 1
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_scaled)[:, 1]
            else:
                y_prob = y_pred # Fallback if model doesn't support probabilities

            # --- Display Evaluation Metrics ---
            st.divider()
            st.subheader(f"📈 Performance Metrics: {selected_model_name}")
            
            # 2. CALCULATE ALL 6 METRICS
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_true, y_pred)

            # 3. DISPLAY ALL 6 METRICS
            # We use 3 columns x 2 rows for a clean look
            
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row2_col1, row2_col2, row2_col3 = st.columns(3)

            # Row 1
            row1_col1.metric("1. Accuracy", f"{acc:.4f}")
            row1_col2.metric("2. AUC Score", f"{auc:.4f}")
            row1_col3.metric("3. Precision", f"{prec:.4f}")
            
            # Row 2 (Added extra padding for visual separation)
            row2_col1.metric("4. Recall", f"{rec:.4f}")
            row2_col2.metric("5. F1 Score", f"{f1:.4f}")
            row2_col3.metric("6. MCC Score", f"{mcc:.4f}")

            # --- Confusion Matrix ---
            st.divider()
            st.subheader(f"Confusion Matrix: {selected_model_name}")
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file in the sidebar to begin evaluation.")
    st.markdown("""
    **CSV Format Requirement:**
    * Must contain the same features columns as the training set.
    * Must contain one Target column (e.g., 'target' or 'diagnosis').
    * **Target Coding:**
        * `0` = Malignant (Cancer)
        * `1` = Benign (Safe)
    """)