import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sklearn.impute import SimpleImputer
import streamlit as st

data = pd.read_csv('customer_churn_data.csv')

# Assuming 'left' is the target column and the rest are input features
cols_to_scale = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
                 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']

# Handling missing values with imputation
imputer = SimpleImputer(strategy='mean')
data[cols_to_scale] = imputer.fit_transform(data[cols_to_scale])

# Check for missing values after imputation
missing_values_after_imputation = data.isnull().sum()
print("Missing values after imputation:")
print(missing_values_after_imputation)

scaler = StandardScaler()
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

X = data.drop(['left', 'empid'], axis=1)
y = data['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Change to KNeighborsClassifier for classification
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    features = [
        data["satisfaction_level"],
        data["last_evaluation"],
        data["number_project"],
        data["average_monthly_hours"],
        data["time_spend_company"],
        data["Work_accident"],
        data["promotion_last_5years"],
        data["salary"]
    ]

    # Ensure that the input features don't contain NaN values
    if any(np.isnan(features)):
        raise HTTPException(status_code=400, detail="Input contains NaN values.")

    features = np.array(features).reshape(1, -1)
    features = imputer.transform(features)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)

    # Extract the probability of the positive class (Employee leaving)
    probability_leave = probabilities[0][1]  # Probability of class 1 (leave)

    # Convert probability to percentage
    probability_percentage = round(probability_leave * 100, 2)

    return JSONResponse(content={"prediction": prediction[0], "probability_leave": f"{probability_percentage} %",
                                 "probabilities": {"stay_probability": f"{round(probabilities[0][0] * 100, 2)} %",
                                                   "leave_probability": f"{round(probabilities[0][1] * 100, 2)} %"}},
                        status_code=200)


st.title("Employee Lifetime Value")
st.write("Enter Employee information:")

satisfaction_level = st.number_input("Satisfaction Level (0-1):", 0.0, 1.0, 0.0)
last_evaluation = st.number_input("Last Evaluation (0-1):", 0.0, 1.0, 0.0)
number_project = st.number_input("Number of Projects:", 0)
average_monthly_hours = st.number_input("Average Monthly Hours:", 0)
time_spend_company = st.number_input("Time Spent in Company (years):", 0)
work_accident = st.checkbox("Has Work Accident?")
promotion_last_5years = st.checkbox("Promoted in Last 5 Years?")
salary = st.selectbox("Salary", ["Low", "Medium", "High"])

salary_dict = {"Low": 0, "Medium": 1, "High": 2}
salary = salary_dict[salary]

X = [satisfaction_level, last_evaluation, number_project, average_monthly_hours,
     time_spend_company, work_accident, promotion_last_5years, salary]
X = np.array(X).reshape(1, -1)
X = scaler.transform(X)

prediction = model.predict(X)
predict_probas = model.predict_proba(X)
if prediction == 0:
    st.write("Employee Likely to stay")
    staying_probability = predict_probas[0, 0]*100
    st.write("Probability of staying: {:.2f}%".format(staying_probability))
else:
    st.write("Employee likely to leave")
    leaving_probability = predict_probas[0, 1]*100
    st.write("Probability of leaving: {:.2f}%".format(leaving_probability))
