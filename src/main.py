import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    data_file = Path("data") / "student_results.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "logistic_regression_results.csv"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Student Result Dataset")
    print("----------------------")
    print(df)

    print()
    print("Result Counts")
    print("-------------")
    print(df["Result"].value_counts())

    X = df[["StudyHours", "Attendance", "AssignmentScore"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "ActualResult": y_test,
        "PredictedResult": predictions
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Accuracy")
    print("--------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

    new_student = pd.DataFrame({
        "StudyHours": [6],
        "Attendance": [78],
        "AssignmentScore": [70]
    })

    new_prediction = model.predict(new_student)
    probabilities = model.predict_proba(new_student)[0]

    print()
    print("New Student Prediction")
    print("----------------------")
    print(new_student)
    print(f"Predicted result: {new_prediction[0]}")

    print()
    print("Prediction Probabilities")
    print("------------------------")

    for class_name, probability in zip(model.classes_, probabilities):
        print(f"{class_name}: {probability * 100:.2f}%")

    results.to_csv(output_file, index=False)

    print()
    print(f"Prediction results saved to: {output_file}")


main()
