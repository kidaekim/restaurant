import pandas as pd

def bootstrap(original_df, sample_size, n_samples):

    frames = [original_df.sample(n=sample_size).assign(sample_number=i) for i in range(n_samples)]
    out_df = pd.concat(frames)
    out_df = out_df.reset_index(drop=True)

    return out_df
