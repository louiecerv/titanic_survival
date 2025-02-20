import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('titanic.csv')

# Title of the app
st.title("Titanic Survival Prediction App")
st.image("titanic_cover.png", caption="Titanic App Cover Image", use_container_width=True)
# About section

with st.expander("About This App"):
    st.header("About This App")
    st.write("""
    This app leverages historical data about the passengers and crew of the Titanic to predict their survival status. The dataset is divided into two parts:

    - **Training set**: Contains information about passengers along with their survival status (survived or not).
    - **Test set**: Contains information about passengers without their survival status, which you need to predict using the model trained on the training set.
    """)

    # Features section
    st.subheader("Features")
    st.write("""
    The dataset includes various features about the passengers, such as:

    - **Passenger class**: Indicates their socioeconomic status (1st, 2nd, 3rd).
    - **Age**: Continuous variable representing the passenger's age at the time of the disaster.
    - **Sex**: Categorical variable indicating the passenger's gender (male or female).
    - **SibSp**: Number of siblings/spouses aboard the Titanic with the passenger.
    - **Parch**: Number of parents/children aboard the Titanic with the passenger.
    - **Ticket**: Ticket number of the passenger.
    - **Fare**: Passenger fare paid in pounds.
    - **Cabin**: Cabin number (sometimes missing).
    - **Embarked**: Port where the passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton).
    """)

    # Target variable section
    st.subheader("Target Variable")
    st.write("""
    The primary target variable is the survival status (**Survived**), indicating whether the passenger survived the disaster (1) or not (0).
    """)

# Preprocessing
# Check for null values and replace with mean for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

# Check for consistent data types for inputs
df = df.convert_dtypes()

# Display the Original DataFrame in Streamlit
st.write("Original DataFrame:")
st.dataframe(df)  # Displays the DataFrame with interactive features (sorting, etc.)

# Show statistics
st.write("Statistics:")
st.write(df.describe())

# List numeric columns and categorical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

st.write("Numeric Columns:", numeric_columns)
st.write("Categorical Columns:", categorical_columns)

# One hot encoding for 'Sex' and 'Embarked' columns
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display the One-hot encoded dataFrame in Streamlit
st.write("One Hot Encoded DataFrame:") 
st.dataframe(df)  

# Example of downloading the DataFrame as a CSV:
@st.cache_data  # Cache the data for faster downloads
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='processed_titanic.csv',
    mime='text/csv',
)

# The last column 'Survived' is the target
target = 'Survived'
X = df.drop(columns=[target])
y = df[target]

# Apply standard scaling for all the columns after encoding
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the scaled DataFrame in Streamlit
st.write("Scaled Training DataFrame (no target):") 
st.dataframe(X_scaled)  

# Label encode the target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset and use 20 percent for testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier()
}

# Create tabs for each model
tabs = st.tabs(list(models.keys()))

for i, (model_name, model) in enumerate(models.items()):
    with tabs[i]:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_encoded = model.predict(X_test)
        
        # Reverse label encoding for predicted values
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_test_decoded, y_pred)
        cr = classification_report(y_test_decoded, y_pred, output_dict=True)
        
        st.write(f"Confusion Matrix for {model_name}:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Show classification report
        st.write("## Classification Report")
        report_df = pd.DataFrame(cr).transpose()
        st.table(report_df)
        
        st.write(f"Remarks on {model_name} performance:")
        st.write("The model's performance can be evaluated based on the classification report and confusion matrix shown above.")