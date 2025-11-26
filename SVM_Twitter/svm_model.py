from modules.loader import load_and_merge
from modules.preprocess import preprocess_dataframe
from modules.vectorizer import build_tfidf
from modules.trainer import train_svm
from modules.evaluation import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# 1 — Carregar bases
full_df = load_and_merge("data/twitter_training.csv",
                         "data/twitter_validation.csv")

# 2 — Ajustar classe Irrelevant → Neutral
print("\nIniciando a substituição da classe 'Irrelevant' para 'Neutral'")
print("\nContagem de sentimentos ANTES da substituição:")
print(full_df['sentiment'].value_counts())

# Remover linhas onde o texto é nulo (Isso evita muitos erros!)
full_df.dropna(subset=["text"], inplace=True)

# Ajuste de classes Irrelevant -> Neutral
full_df["sentiment"] = full_df["sentiment"].replace({"Irrelevant": "Neutral"})

print("\nContagem de sentimentos APÓS substituição e remoção de linhas com texto nulo:")
print(full_df['sentiment'].value_counts())

# 3 — Pré-processamento
full_df = preprocess_dataframe(full_df)

print("\n📊 Estatísticas pós-pré-processamento:")
print(full_df['sentiment'].value_counts())

# 4 — Salvar pré-processado
full_df.to_csv("data/twitter_data_preprocessed.csv", index=False)
print(f"\n💾 Base pré-processada salva como 'data/twitter_data_preprocessed.csv'")

# 5 — Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    full_df["cleaned_text"],
    full_df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=full_df["sentiment"]
)

# 6 — Vetorização
tfidf = build_tfidf()

print("\n🔠 Vetorizando textos...")
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 7 — Treino
print("\n🔍 Iniciando treino do modelo com GridSearch do SVM...")
print("⚠️ Aviso: O SVM é lento com muitos dados. Se demorar demais, considere reduzir o dataset.")
model = train_svm(X_train_vec, y_train)


# 8 — Avaliação
evaluate_model(model, X_test_vec, y_test)

print("\n🎉 Fim da Execução")
