import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Configurações NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TweetTokenizer()

# Dicionário de gírias
slang_dict = {
    "u": "you", "ur": "your", "pls": "please", "plz": "please",
    "thx": "thanks", "imo": "in my opinion", "idk": "i don't know",
    "btw": "by the way", "omg": "oh my god", "lol": "laughing",
    "lmao": "laughing", "rofl": "laughing", "tbh": "to be honest",
    "bc": "because", "bcz": "because", "brb": "be right back",
    "gr8": "great", "smh": "shaking my head"
}

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def handle_negation(tokens):
    negations = {"not", "no", "never", "n't"}
    new_tokens = []
    neg_flag = False

    for token in tokens:
        if token in negations:
            neg_flag = True
            continue
        if neg_flag:
            new_tokens.append("not_" + token)
            neg_flag = False
        else:
            new_tokens.append(token)

    return new_tokens

def handle_intensifiers(tokens):
    intensifiers = {"very", "extremely", "too", "so", "really"}
    new_tokens = []
    skip = False

    for i in range(len(tokens)):
        if skip:
            skip = False
            continue

        if i < len(tokens) - 1 and tokens[i] in intensifiers:
            new_tokens.append(tokens[i] + "_" + tokens[i + 1])
            skip = True
        else:
            new_tokens.append(tokens[i])

    return new_tokens


# ---------------------------------------------------------
# PREPROCESSAMENTO DE TEXTO INDIVIDUAL
# ---------------------------------------------------------

def preprocess_tweet(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = replace_emojis(text)
    text = contractions.fix(text)

    for slang, full in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', full, text)

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = tokenizer.tokenize(text)
    tokens = handle_negation(tokens)
    tokens = handle_intensifiers(tokens)
    tokens = [t for t in tokens if len(t) > 1]

    tagged = pos_tag(tokens)
    processed = [
        lemmatizer.lemmatize(w, get_wordnet_pos(tag))
        for w, tag in tagged if w not in stop_words
    ]

    return " ".join(processed)


# ---------------------------------------------------------
# PREPROCESSAMENTO DO DATAFRAME COMPLETO
# ---------------------------------------------------------

def preprocess_dataframe(df):
    print("\n🧼 Aplicando pré-processamento em todos os tweets...")

    df["cleaned_text"] = df["text"].apply(preprocess_tweet)

    print("   ✔ Pré-processamento concluído.")
    print("   ✔ Removendo linhas vazias após pré-processamento...")

    # Remoção robusta de linhas vazias
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    df = df[
        df["cleaned_text"]
            .str.strip()
            .replace(r'\s+', '', regex=True)
            .astype(bool)
    ]

    print(f"   ✔ Linhas válidas restantes: {len(df)}")
    return df
