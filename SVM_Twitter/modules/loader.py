import pandas as pd

def load_and_merge(training_path, validation_path, save_path="data/twitter_full_raw.csv"):
    print("📥 Carregando arquivos...")

    train_df = pd.read_csv(training_path, header=None,
                           names=['tweet_id', 'entity', 'sentiment', 'text'])

    val_df = pd.read_csv(validation_path, header=None,
                         names=['tweet_id', 'entity', 'sentiment', 'text'])

    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\nRegistros totais: {len(full_df)}")

    full_df.to_csv(save_path, index=False)
    print(f"\n💾 Base unificada salva como '{save_path}'")
    return full_df
