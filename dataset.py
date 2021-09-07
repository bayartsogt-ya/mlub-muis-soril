from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=150, ):
        self.data = []
        for row in tqdm(df.itertuples(), total=df.shape[0]):
            encoding = tokenize_mlub(row.text_truncated, tokenizer, max_len=max_len)
            # encoding = {k:v[0] for k, v in encoding.items()}
            encoding["labels"] = torch.tensor(row.synset_index, dtype=torch.long)
            # del encoding["start_end_mask"]
            self.data.append(encoding)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def tokenize_mlub(text, tokenizer, max_len):
    input_ids = []
    attention_mask = []
    start = end = None
    for index, tok in enumerate(text.split()):
        if "#" in tok:
            tok = tok.split("#")[0]
            start = len(input_ids)
            end = len(input_ids) + len(tokenizer.encode(tok))
        input_ids += tokenizer.encode(tok + " ")
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_len:
        raise Exception("input_ids is longer than max_len, please increase max_len")
    
    attention_mask += [0] * (max_len - len(input_ids))

    if tokenizer.is_fast and tokenizer.pad_token_id < tokenizer.vocab_size - 1:
        padding_id = tokenizer.pad_token_id
    else:
        padding_id = tokenizer.eos_token_id

    if "gpt" in tokenizer.name_or_path:
        padding_id = 1 # which is <pad>
    if "tugstugi" in tokenizer.name_or_path or "mlub" in tokenizer.name_or_path:
        padding_id = 3
    input_ids.extend([padding_id] * (max_len - len(input_ids)))

    start_end_mask = [0] * max_len
    for i in range(start, end):
        start_end_mask[i] = 1

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "start_end_mask": torch.tensor(start_end_mask, dtype=torch.long),
    }

def truncate_text(df, effective_len = 30):
    list_truncated_text = []
    for row in df.itertuples():
        # get query token location
        tokens = row.text.split()
        synset_query = [(tok, index) for index, tok in enumerate(tokens) if "#" in tok]; assert len(synset_query) == 1
        query_word, query_location = synset_query[0]
        
        # truncate
        begin = max(query_location - effective_len//2, 0)
        end   = min(query_location + effective_len//2+1, len(tokens))
        tokens = tokens[begin: end]

        # store
        list_truncated_text.append(" ".join(tokens))

    df["text_truncated"] = list_truncated_text
    return df

def get_index2synsetid_dicts(synset_word, dict_synset_meaning):
    """index2id, id2index = get_index2synsetid_dicts("ам")"""
    # construct index to id dictionaries
    ids = sorted(list(dict_synset_meaning[synset_word].keys()))
    return {i:index for i, index in enumerate(ids)}, {index:i for i, index in enumerate(ids)}
