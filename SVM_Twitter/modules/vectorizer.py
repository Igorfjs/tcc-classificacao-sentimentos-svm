from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(max_features=10000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        stop_words="english"
    )
