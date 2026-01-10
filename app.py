import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
model_options = ["Logistic Regression"]
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
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target # 0=Malignant, 1=Benign
        
        st.info("✅ Using built-in Breast Cancer Wisconsin dataset.")
        
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
            
            # Make Predictions
            y_pred = model.predict(X_scaled)
            
            # --- Display Evaluation Metrics ---
            st.divider()
            st.subheader(f"📈 Performance Metrics: {selected_model_name}")
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1 Score", f"{f1:.4f}")

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