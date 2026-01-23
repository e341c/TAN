import numpy as np

def check_no_leakage(X_seq, df, target_feature="log_return", features_list=None, seq_len=30):
    """
    Checks that for every sequence in X_seq, the prediction target (t+seq_len) is not inside the sequence.
    
    Args:
        X_seq: np.array of shape (n_sequences, seq_len, n_features)
        df: original dataframe with target column
        target_feature: name of the column we are predicting
        features_list: list of feature names corresponding to columns in X_seq
        seq_len: length of each sequence
    Returns:
        True if no leakage detected, False otherwise
    """
    if features_list is None:
        raise ValueError("features_list must be provided")
    
    target_idx = features_list.index(target_feature)
    n_sequences = X_seq.shape[0]
    leakage_found = False
    
    for i in range(n_sequences):
        seq = X_seq[i]
        target_val = df[target_feature].values[i + seq_len]  # the value model should predict
        if target_val in seq[:, target_idx]:
            print(f"⚠️ Leakage detected in sequence {i}: target value {target_val} appears in input sequence!")
            leakage_found = True
    
    if not leakage_found:
        print("✅ No leakage detected in any sequence.")
        return True
    else:
        return False
