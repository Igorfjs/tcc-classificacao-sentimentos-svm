# 📊 Classificação de Sentimentos em Tweets

Este projeto implementa um sistema completo de **classificação de sentimentos** aplicado a tweets, utilizando diversas técnicas de Processamento de Linguagem Natural (PLN) e múltiplos modelos de machine learning:

- **SVM (TF-IDF)**
- **Regressão Logística (TF-IDF)**
- **Naive Bayes e Complement Naive Bayes (TF-IDF otimizado)**

O objetivo principal é comparar esses modelos usando métricas clássicas de avaliação e entender qual abordagem oferece o melhor desempenho para sentimento **positivo**, **negativo** e **neutro**.

O modelo SVM é o principal modelo estudado nesse projeto. Por isso, o módulo svm_model também cria e formata os arquivos de base de dados e deve ser rodado primeiro em outros computadores.
---

## 📁 Estrutura do Projeto

├── data/
│ ├── twitter_training.csv
│ ├── twitter_validation.csv
│ ├── twitter_full_raw.csv
│ ├── twitter_data_preprocessed.csv
│
├── modules/
│ ├── preprocess.py
│ ├── data_loader.py
│ ├── models.py
│ ├── evaluation.py
│ ├── utils.py
│ └── init.py
│
├── svm_main.py
├── logistic_regression_main.py
├── naive_bayes_main.py
├── neural_network_main.py
│
├── requirements.txt
└── README.md


---

## 📥 Bases de Dados

O projeto utiliza as bases originais retiradas de https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data:

- **twitter_training.csv**
- **twitter_validation.csv**

E gera automaticamente:

- `twitter_full_raw.csv` → bases unificadas  
- `twitter_data_preprocessed.csv` → textos pré-processados

Esses arquivos são usados por todos os modelos posteriores.

---

## 🧼 Pré-processamento

O pipeline de pré-processamento inclui:

- conversão para minúsculas  
- remoção otimizada de URLs, hashtags e menções  
- normalização de espaços  
- substituição de emojis → palavras (`emoji.demojize`)  
- expansão de contrações (`contractions`)  
- substituição de gírias (slang dictionary manual)  
- tokenização com `TweetTokenizer`  
- preservação de negações (`not_word`)  
- tratamento de intensificadores (`very_good → very_good`)  
- POS-tagging + lematização  
- remoção de stopwords  
- remoção robusta de linhas vazias  

Esse pré-processamento foi modularizado no arquivo **preprocess.py**.

---

## 🧪 Modelos Implementados

### ✔ 1. **SVM com TF-IDF**
- TF-IDF com n-grams (1,2)
- `max_features=10000`
- `min_df=5`, `max_df=0.7`
- GridSearch com SVM Linear e RBF

---

### ✔ 2. **Regressão Logística com TF-IDF**
- Solvers testados: `lbfgs`, `saga`
- Ajuste de hiperparâmetros (`C`, regularização)
- Ideal como baseline forte

---

### ✔ 3. **Naive Bayes**
- MultinomialNB
- Complement Naive Bayes (melhor para dados desbalanceados)
- TF-IDF com `sublinear_tf=True`
- GridSearch de `alpha`


## 📈 Métricas de Avaliação

Todos os modelos geram:

- **Accuracy**
- **Macro F1-score**
- **Classification report**
- **Matriz de Confusão**
- **Curva ROC (Multiclasse – One vs Rest)**
- **AUC para cada classe**

Os gráficos são exibidos automaticamente e podem ser salvos.

---

## 🗂 Salvamento de Arquivos

O projeto salva:

- `twitter_full_raw.csv`  
- `twitter_data_preprocessed.csv`  
- gráfico da matriz de confusão  
- curva ROC  
- resultados das métricas  
- modelos treinados (`joblib.dump`)  
- TF-IDF (`joblib.dump`)  

---

## ▶ Execução dos Modelos

### SVM:
Rodar o módulo svm_model.py

---

### Naive Bayes:
Rodar o módulo naive_bayes_model.py

---

### Regressão Linear:
Rodar o módulo logistic_regression_model.py


---

## ⚙ Tecnologias Utilizadas

- Python 3.10+
- pandas, numpy
- scikit-learn
- nltk
- emoji
- contractions
- seaborn, matplotlib
- joblib

---

## 🧠 Possíveis Melhorias Futuras

- Fine-tuning com **BERTweet**, modelo específico para Twitter  
- Aumento da base de dados (data augmentation)  
- Aplicar SMOTE ou técnicas de balanceamento  
- Ajuste avançado de hiperparâmetros com Optuna  
- Converter o modelo final para API (FastAPI ou Flask)  
- Implementar front-end para demonstração  
- Utilização de GPU para execução
- Implementaçar um modelo de redes neurais

---

## 👨‍💻 Autor

Projeto desenvolvido para fins acadêmicos e experimentação de modelos de machine learning aplicados a análise de sentimentos em textos curtos (Tweets).

---


