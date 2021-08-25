import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

def get_kfold(df, num_folds=5, random_state=42):
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(kfold.split(X=df.text_id, y=df.synset_id)):
        df.loc[valid_idx, "fold"] = fold
    return df

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    f1 = f1_score(pred_ids, pred.label_ids, average="macro")
    acc = accuracy_score(pred_ids, pred.label_ids)
    return {"f1": f1, "accuracy":acc}