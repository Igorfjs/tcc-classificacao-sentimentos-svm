import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import ComplementNB, MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ==============================
# CONFIGURAÇÕES
# ==============================
DATA_PREPROCESSED = "data/twitter_data_preprocessed.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_CONFIG = dict(
    max_features=50000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
    stop_words=None
)

ALPHAS = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

CV = 5
N_JOBS = -1


# ==============================
# FUNÇÕES
# ==============================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    if "cleaned_text" not in df.columns:
        raise ValueError("CSV não contém coluna 'cleaned_text'.")

    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df = df[df["cleaned_text"].str.strip().astype(bool)]

    return df.reset_index(drop=True)


# ==============================
# PIPELINE PRINCIPAL
# ==============================

def main():
    print("📥 Carregando dados...")
    df = load_data(DATA_PREPROCESSED)
    print(f"Registros: {len(df)}")

    X = df["cleaned_text"]
    y = df["sentiment"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Vetorização
    print("\n🔠 TF-IDF otimizado...")
    tfidf = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    labels = sorted(df["sentiment"].unique())

    # ==============================
    # ComplementNB
    # ==============================
    print("\n🔎 GridSearch — ComplementNB")
    cnb = ComplementNB()

    grid_cnb = GridSearchCV(
        cnb,
        {"alpha": ALPHAS},
        scoring="accuracy",
        cv=CV,
        n_jobs=N_JOBS,
        verbose=1
    )

    grid_cnb.fit(X_train_vec, y_train)
    best_cnb = grid_cnb.best_estimator_

    print("→ Melhor alpha ComplementNB:", grid_cnb.best_params_)

    # Previsões
    y_pred_cnb = best_cnb.predict(X_test_vec)

    print("\n==============================")
    print("📊 RESULTADOS — ComplementNB")
    print("==============================")
    print("Acurácia:", accuracy_score(y_test, y_pred_cnb))
    print("\nRelatório:\n", classification_report(y_test, y_pred_cnb))

    # Matriz de confusão CNB
    cm_cnb = confusion_matrix(y_test, y_pred_cnb)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_cnb, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusão - ComplementNB")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    # Curva ROC CNB
    try:
        y_prob_cnb = best_cnb.predict_proba(X_test_vec)
        y_test_bin = label_binarize(y_test, classes=labels)

        plt.figure(figsize=(8, 6))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_cnb[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("Curva ROC - ComplementNB")
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.legend()
        plt.show()

    except Exception as e:
        print("⚠️ Falha ao gerar ROC para ComplementNB:", e)

    # ==============================
    # MultinomialNB
    # ==============================
    print("\n🔎 GridSearch — MultinomialNB")
    mnb = MultinomialNB()

    grid_mnb = GridSearchCV(
        mnb,
        {"alpha": ALPHAS},
        scoring="accuracy",
        cv=CV,
        n_jobs=N_JOBS,
        verbose=1
    )

    grid_mnb.fit(X_train_vec, y_train)
    best_mnb = grid_mnb.best_estimator_

    print("→ Melhor alpha MultinomialNB:", grid_mnb.best_params_)

    # Previsões
    y_pred_mnb = best_mnb.predict(X_test_vec)

    print("\n==============================")
    print("📊 RESULTADOS — MultinomialNB")
    print("==============================")
    print("Acurácia:", accuracy_score(y_test, y_pred_mnb))
    print("\nRelatório:\n", classification_report(y_test, y_pred_mnb))

    # Matriz de confusão MNB
    cm_mnb = confusion_matrix(y_test, y_pred_mnb)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_mnb, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusão - MultinomialNB")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    # Curva ROC MNB
    try:
        y_prob_mnb = best_mnb.predict_proba(X_test_vec)
        y_test_bin = label_binarize(y_test, classes=labels)

        plt.figure(figsize=(8, 6))
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_mnb[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("Curva ROC - MultinomialNB")
        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.legend()
        plt.show()

    except Exception as e:
        print("⚠️ Falha ao gerar ROC para MultinomialNB:", e)

    print("\n✅ Execução finalizada!")


if __name__ == "__main__":
    main()
