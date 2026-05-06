from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(y_test, y_pred):
    print("\nModel Evaluation:")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
    print("Recall:", recall_score(y_test, y_pred, pos_label="spam"))
    print("F1 Score:", f1_score(y_test, y_pred, pos_label="spam"))
    print("=" * 60)


def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    print("\nConfusion Matrix:")
    print("=" * 60)
    print(f"{'':<12}{'|Pred: ham':<15}{'Pred: spam':<15}")
    print(f"{'Actual: ham |':<12}{cm[0][0]:<15}{cm[0][1]:<15}")
    print(f"{'Actual: spam|':<12}{cm[1][0]:<15}{cm[1][1]:<15}")
    print("=" * 60)
