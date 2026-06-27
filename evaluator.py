from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import mlflow


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print(f"🎯 Accuracy : {accuracy * 100:.2f}%")
    print(f"📊 Precision : {precision * 100:.2f}%")
    print(f"📊 Recall : {recall * 100:.2f}%")
    print(f"📊 F1 Score : {f1 * 100:.2f}%")
    print("\n📊 Classification Report :")
    print(classification_report(y_test, y_pred))

    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    print("📁 Matrice sauvegardée dans confusion_matrix.png")

    return accuracy
