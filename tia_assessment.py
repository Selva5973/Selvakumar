import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# Load your dataset
def load_data():
    # Assuming your dataset is a CSV file named 'tia_data.csv'
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return data

# Data preprocessing function
def preprocess_data(data):
    data = data.drop(columns=['id'])  # Drop 'id' column
    data = data.dropna() 

    # Encode categorical variables
    label_encoders = {}
    for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
     


    # Features and target variable
    features = data.drop('stroke', axis=1)
    labels = data['stroke']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

# Define LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load Data
data = load_data()

# Preprocess Data
X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(data)

# Reshape data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Train Logistic Regression Model with Hyperparameter Tuning
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Train SVM Model with Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid_search.fit(X_train, y_train)
svm_model = grid_search.best_estimator_

# Train LSTM Model
lstm_model = create_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=16, verbose=0, callbacks=[early_stopping])

# Evaluate models
def evaluate_models(X_test, y_test):
    # Logistic Regression evaluation
    log_pred = logistic_model.predict(X_test)
    log_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
    st.write("### Logistic Regression Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, log_pred):.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, log_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, log_pred))

    # SVM evaluation
    svm_pred = svm_model.predict(X_test)
    svm_pred_proba = svm_model.predict_proba(X_test)[:, 1]
    st.write("### SVM Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, svm_pred):.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, svm_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, svm_pred))

    # LSTM evaluation
    lstm_pred = lstm_model.predict(X_test_lstm)
    lstm_pred_class = (lstm_pred > 0.5).astype(int)
    st.write("### LSTM Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, lstm_pred_class):.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, lstm_pred_class))
    st.write("Classification Report:")
    st.text(classification_report(y_test, lstm_pred_class))

# Streamlit UI for User Input
st.title("TIA Risk Assessment Application")

# Display the distribution of the target variable
st.subheader("Data Distribution")
sns.countplot(data=data, x='stroke')
plt.title("Distribution of Stroke")
plt.xticks([0, 1], ["No Stroke", "Stroke"])
st.pyplot(plt)

# Collect user input
st.write("### Input Patient Health Metrics")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "smokes", "formerly smoked"])

# Prepare input for models
user_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

# Encode the user input
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    user_data[0, column] = label_encoders[column].transform([user_data[0, column]])[0]

# Scale the user input
user_data_scaled = scaler.transform(user_data)

# Make predictions
if st.button("Predict TIA Risk"):
    # Logistic Regression Prediction
    log_pred = logistic_model.predict(user_data_scaled)
    log_pred_proba = logistic_model.predict_proba(user_data_scaled)

    # SVM Prediction
    svm_pred = svm_model.predict(user_data_scaled)
    svm_pred_proba = svm_model.predict_proba(user_data_scaled)

    # Prepare data for LSTM prediction
    user_data_lstm = user_data_scaled.reshape((1, 1, user_data_scaled.shape[1]))
    lstm_pred = lstm_model.predict(user_data_lstm)
    lstm_pred_class = 1 if lstm_pred > 0.5 else 0

    # Display predictions
    st.write("### Predictions:")
    st.write(f"Logistic Regression Prediction: {'High risk' if log_pred[0] == 1 else 'Low risk'}, Probability: {log_pred_proba[0][1]:.2f}")
    st.write(f"SVM Prediction: {'High risk' if svm_pred[0] == 1 else 'Low risk'}, Probability: {svm_pred_proba[0][1]:.2f}")
    st.write(f"LSTM Prediction: {'High risk' if lstm_pred_class == 1 else 'Low risk'}, Probability: {lstm_pred[0][0]:.2f}")

# Evaluate models after predictions (optional)
evaluate_models(X_test, y_test)
