import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Load Wine dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train-Test SPlit
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

#Define the parameters for RF Model
max_depth = 10
n_estimators = 5

# Mention or set experiment below
mlflow.set_experiment("Experiment-from-UI")

# You can also set the experiment inside the strt_run method
################# with mlflow.start_run(experiment_id=966970494872304366):  ###################

# create a new exepriment from the code
mlflow.set_experiment("Experiment-from-code")


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    print(accuracy)

    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the confusion matrix
    plt.savefig("Confusion_Matrix.png")

    # Log the confusion matrix as an artifact
    mlflow.log_artifact("Confusion_Matrix.png")
    mlflow.log_artifact(__file__)

    # Add tags
    mlflow.set_tags({"Author": 'Naman', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")