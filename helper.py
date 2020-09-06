import pandas as pd

def bootstrap(original_df, sample_size, n_samples):

    frames = [original_df.sample(n=sample_size).assign(sample_number=i) for i in range(n_samples)]
    out_df = pd.concat(frames)
    out_df = out_df.reset_index(drop=True)

    return out_df

def train_val_split(original_df, val_size, seed):

    val_df = original_df.sample(n=val_size, random_state=seed)
    train_df = original_df.drop(val_df.index, axis=0)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    return train_df, val_df
