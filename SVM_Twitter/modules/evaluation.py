import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n📄 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("\n🎯 Acurácia:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test)))

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Matriz de Confusão - SVM")
    plt.show()

    try:
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=labels)
        plt.figure(figsize=(7, 6))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("Curva ROC – Multiclasse (SVM)")
        plt.legend()
        plt.show()
    except:
        print("⚠️ ROC não pôde ser gerada (algum erro)")
