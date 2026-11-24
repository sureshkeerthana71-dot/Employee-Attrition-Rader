# Employee Attrition Radar

## 1. Project Overview

Employee Attrition Radar is a Data Science and Machine Learning project that analyses employee attrition and identifies patterns associated with employees leaving an organisation.

The project helps HR teams understand employee attrition and explore possible retention strategies.

## 2. Business Problem

The HR department wants to understand:

- Why are employees leaving?
- Which employee groups have higher attrition rates?
- Is overtime associated with employee attrition?
- Does job satisfaction affect attrition?
- Can Machine Learning predict employee attrition?

## 3. Dataset

Dataset: IBM HR Analytics Employee Attrition & Performance.

Source: Kaggle.

The dataset contains approximately 1,470 employee records and 35 columns.

The dataset is fictional and intended for analytics and educational purposes.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 5. Project Workflow

1. Dataset collection.
2. Data cleaning.
3. Exploratory Data Analysis.
4. Data preprocessing.
5. Training and testing split.
6. Logistic Regression model.
7. Random Forest model.
8. Model evaluation.
9. Feature importance analysis.
10. HR recommendations.

## 6. Machine Learning Models

Two algorithms are compared:

- Logistic Regression
- Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Recall is given special attention because identifying employees who actually leave is important.

## 7. Key Findings

Add your actual results after running the application.

- Overall attrition rate: [Enter result]
- Overtime attrition rate: [Enter result]
- Highest attrition job role: [Enter result]
- Best-performing model: [Enter result]
- Recall: [Enter result]

## 8. HR Recommendations

1. Monitor overtime and employee workload.
2. Improve job satisfaction.
3. Provide career growth and employee support.

These recommendations should be updated based on the actual analysis.

## 9. How to Run the Project

Install the required libraries:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## 10. What I Verified and What I Changed by Hand

- Verified the overall attrition rate independently using Excel.
- Verified the overtime attrition rate using an Excel PivotTable.
- Verified the employee group attrition rates using an Excel PivotTable.
- Checked the training and testing split.
- Ensured that preprocessing was fitted only on training data.
- Checked the confusion matrix and recall values.

Add your actual Excel calculations and verified results here.

## 11. Ethical Considerations

This project must not be used to punish, discriminate against or single out individual employees.

The model should only be used to understand general employee attrition patterns and support improvements in workplace conditions.

Predictions are not proof that an employee will leave.

Human judgement, employee feedback and privacy must always be respected.

## 12. Limitations

- The dataset is fictional.
- The dataset may not represent every organisation.
- Machine Learning predictions can be incorrect.
- Correlation does not prove causation.
- The model should not make employment decisions.

## 13. Conclusion

Employee Attrition Radar uses Data Science and Machine Learning to understand employee attrition patterns.

It compares two classification algorithms and provides information that can help HR teams explore employee retention strategies.