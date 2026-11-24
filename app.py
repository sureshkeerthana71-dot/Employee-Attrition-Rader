import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

st.title("Employee Attrition Radar")

st.write("Employee Attrition Analysis Dashboard")

st.subheader("Employee Dataset")

st.dataframe(df.head())
st.write("Dataset Shape:", df.shape)

st.write("Number of Rows:", df.shape[0])

st.write("Number of Columns:", df.shape[1])

st.write("Missing Values:")

st.write(df.isnull().sum())

st.write("Duplicate Rows:")

st.write(df.duplicated().sum())
df = df.drop_duplicates()

df = df.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

st.subheader("Cleaned Dataset")

st.write(df.head())

st.write("Missing Values:", df.isnull().sum().sum())
total_employees = len(df)

employees_left = (
    df["Attrition"] == "Yes"
).sum()

attrition_rate = (
    employees_left / total_employees
) * 100

st.subheader("Overall Attrition")

st.metric(
    "Attrition Rate",
    f"{attrition_rate:.2f}%"
)

st.metric(
    "Employees Left",
    employees_left
)

st.metric(
    "Total Employees",
    total_employees
)
st.subheader("Attrition by Overtime")

overtime_data = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

st.bar_chart(overtime_data)
st.subheader("Attrition by Job Role")

job_data = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
) * 100

st.bar_chart(job_data)
st.subheader("Attrition by Job Satisfaction")

satisfaction_data = df.groupby(
    "JobSatisfaction"
)["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(satisfaction_data)
st.subheader("Attrition by Years at Company")

df["TenureGroup"] = pd.cut(
    df["YearsAtCompany"],
    bins=[-1, 2, 5, 10, 50],
    labels=[
        "0-2 Years",
        "3-5 Years",
        "6-10 Years",
        "Above 10 Years"
    ]
)

tenure_data = df.groupby(
    "TenureGroup",
    observed=False
)["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(tenure_data)
st.subheader("Attrition by Income")

df["IncomeGroup"] = pd.cut(
    df["MonthlyIncome"],
    bins=3,
    labels=[
        "Low Income",
        "Medium Income",
        "High Income"
    ]
)

income_data = df.groupby(
    "IncomeGroup",
    observed=False
)["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(income_data)
st.subheader("Top Attrition Groups")

job_rates = (
    df.groupby("JobRole")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

top_jobs = job_rates.sort_values(
    ascending=False
).head(3)

st.write("Top 3 Job Roles")

st.dataframe(top_jobs)
overtime_rates = (
    df.groupby("OverTime")["Attrition"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

st.write("Overtime Attrition Rates")

st.dataframe(overtime_rates)
st.write("Tenure Attrition Rates")

st.dataframe(tenure_data.sort_values(
    ascending=False
))
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})
X = df.drop(
    columns=["Attrition", "TenureGroup", "IncomeGroup"],
    errors="ignore"
)

y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_columns
        ),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)
random_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2
            )
        )
    ]
)

random_model.fit(X_train, y_train)
logistic_prediction = logistic_model.predict(X_test)

random_prediction = random_model.predict(X_test)
def evaluate_model(model_name, actual, predicted):

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            actual, predicted
        ),

        "Precision": precision_score(
            actual, predicted,
            zero_division=0
        ),

        "Recall": recall_score(
            actual, predicted,
            zero_division=0
        ),

        "F1 Score": f1_score(
            actual, predicted,
            zero_division=0
        )
    }


results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_prediction
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        random_prediction
    )
)

results_df = pd.DataFrame(results)

st.subheader("Model Comparison")

st.dataframe(results_df)
st.subheader("Random Forest Confusion Matrix")

cm = confusion_matrix(
    y_test,
    random_prediction
)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"],
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

plt.close(fig)
st.subheader("Top Factors Behind Attrition")

importance = permutation_importance(
    random_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="recall"
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df.head(10))

top_features = importance_df.head(10)

fig, ax = plt.subplots()

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    ax=ax
)

st.pyplot(fig)

plt.close(fig)
st.subheader("Recommendations for HR")

st.markdown("""
### Recommendation 1: Monitor Overtime

Check whether employees working overtime
have higher attrition rates.

Review workloads and working hours.

### Recommendation 2: Improve Job Satisfaction

Collect employee feedback regularly.

Identify workplace problems and improve
employee satisfaction.

### Recommendation 3: Improve Employee Retention

Review career growth opportunities,
promotion policies and employee support.

Provide regular feedback and development plans.
""")
st.subheader("What-If Analysis")

base_profile = X.mode().iloc[[0]].copy()

overtime_choice = st.selectbox(
    "Select Overtime",
    ["Yes", "No"]
)

satisfaction_choice = st.slider(
    "Job Satisfaction",
    min_value=1,
    max_value=4,
    value=3
)

base_profile["OverTime"] = overtime_choice

base_profile["JobSatisfaction"] = satisfaction_choice

probability = random_model.predict_proba(
    base_profile
)[0][1]

st.metric(
    "Predicted Attrition Probability",
    f"{probability * 100:.2f}%"
)
import joblib

joblib.dump(
    random_model,
    "attrition_model.pkl"
)
model = joblib.load("attrition_model.pkl")
