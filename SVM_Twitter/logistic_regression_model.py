# ======================================================================================
# CLASSIFICAÇÃO DE SENTIMENTOS USANDO REGRESSÃO LOGÍSTICA (TF-IDF)
# COMPARAÇÃO COM O MODELO SVM
# ======================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ======================================================================================
# 1. CARREGAMENTO DAS BASES
# ======================================================================================

print("\n📥 Carregando dados unificados e pré-processados...")

try:
    raw_df = pd.read_csv("data/twitter_full_raw.csv")
    full_df = pd.read_csv("data/twitter_data_preprocessed.csv")
except FileNotFoundError:
    print("❌ Arquivos não encontrados. Certifique-se de que twitter_full_raw.csv e twitter_data_preprocessed.csv existem.")
    exit()

print("✔ Bases carregadas com sucesso!")
print(f"Total de registros carregados: {len(full_df)}")

# ======================================================================================
# 2. DIVISÃO TREINO/TESTE
# ======================================================================================

X = full_df["cleaned_text"]
y = full_df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n📊 Divisão treino/teste concluída:")
print(f"Treino: {len(X_train)} registros")
print(f"Teste:  {len(X_test)} registros")

# ======================================================================================
# 3. TF-IDF
# ======================================================================================

print("\n🔠 Vetorizando textos com TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("✔ Vetorização concluída!")
print(f"Dimensão dos vetores: {X_train_vec.shape}")

# ======================================================================================
# 4. REGRESSÃO LOGÍSTICA
# ======================================================================================

print("\n🤖 Treinando modelo de Regressão Logística...")

logreg = LogisticRegression(
    max_iter=300,
    n_jobs=-1,
    class_weight="balanced",
    solver="saga",
    multi_class="auto"
)

logreg.fit(X_train_vec, y_train)

print("✔ Modelo treinado!")

# ======================================================================================
# 5. AVALIAÇÃO DO MODELO
# ======================================================================================

y_pred = logreg.predict(X_test_vec)

print("\n📈 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Acurácia: {acc:.4f}")

# ======================================================================================
# 6. MATRIZ DE CONFUSÃO
# ======================================================================================

labels = sorted(full_df["sentiment"].unique())
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão – Logistic Regression")
plt.tight_layout()
plt.show()

# ======================================================================================
# 7. CURVA ROC MULTICLASSE
# ======================================================================================

print("\n📉 Gerando Curva ROC (multiclasse)...")

try:
    y_prob = logreg.predict_proba(X_test_vec)
    y_test_bin = label_binarize(y_test, classes=labels)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdadeiros Positivos")
    plt.title("Curva ROC – Logistic Regression (Multiclasse)")
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"⚠️ Não foi possível gerar a curva ROC: {e}")

print("\n✅ Execução concluída com sucesso!")
