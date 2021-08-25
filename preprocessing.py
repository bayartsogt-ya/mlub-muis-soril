import gc
import os
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


replace_by_empty = [
    "...", "..", "\",", '"', '%', '&', "'", '(', ')',
    '*', '+', '-', '/', ':', ';', '<', '>',
    '[', '\\', ']', '{', '}', '¬', '‘', '’', '№', "r"
]

replace_by_space = [
    '\n', 
]

def preprocess(text:str):
    text = text.lower()
    for c in replace_by_empty:
        text = text.replace(c, "")
    for c in replace_by_space:
        text = text.replace(c, " ")
    return text


def preprocess_test(text:str):
    for c in ["...", "..", " ", "...\"", "!\""]:
        text = text.replace(c+"#", "#")
    
    return preprocess(text) # delegate preprocess
    

def preprocess_consecutive_item_error(df: pd.DataFrame, intersection_threshold=0.5):
    """
    images/ss0.png-ээр харахад дараалалсан дата дотор цэвэрлэгээгээ буруу хийснээс болж гаралтууд хоорондоо нийлсэн байгааг
    reverse engineer хийх
    """
    new_texts = []
    latest_similar = "random " * 200
    potential_bad_dict = {}
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text
        tokens_prev = latest_similar.split()
        tokens_cur  = text.split()

        intersection = set(tokens_cur).intersection(tokens_prev)
        union = set(tokens_cur).union(tokens_prev)

        if len(intersection) / len(union) > intersection_threshold:
            keys = list(potential_bad_dict.keys())
            keys.reverse()

            new_text = text
            for key in keys:
                new_text =new_text.replace(key, " ".join(potential_bad_dict[key]))
            new_texts.append((i, row.text_id, new_text))
            
            df.loc[i, "text"] = new_text #     <------- Replacing
        else:
            potential_bad_dict = {}
            latest_similar = text

        query_word = None
        for tok in tokens_cur:
            if "#" in tok:
                query_word = tok.split("#")[0]
                continue
            if query_word:
                potential_bad_dict[query_word + tok] = [query_word, tok]
                break
    
    return df

def replace_labeling_error(df, index=101, old="улаан,мөнгө", new="улаан, мөнгө"):
    """
    Датаны зарим хэсэг алдаатай label-дсан байсны in-place засах
    """
    df.loc[index, "text"] = df.loc[index, "text"].replace(old, new)


def calculate_similar_by_start(token1:str, token2:str) -> int:
    count = 0
    for i in range(min(len(token1), len(token2))):
        if token1[i] == token2[i]:
            count += 1
        else:
            break
    return count / min(len(token1), len(token2))

def find_closest(query_token, unique_synset):
    if query_token in unique_synset:
        return query_token, 1.
    
    best = float("-inf")
    best_syn = None
    for syn in unique_synset:
        d = calculate_similar_by_start(query_token, syn)
        if d > best:
            best_syn = syn
            best = d
            
    return best_syn, best

def find_closest_meaning(text, best_syn, dict_synset_meaning):
    text = handle_special_char(text) # handling "#id"
    
    best = float("-inf")
    best_idx = None
    best_meaning = None
    for idx, meaning in dict_synset_meaning[best_syn].items():
        d = count_intersections(text, preprocess(meaning))
        if d > best:
            best_idx = idx
            best_meaning = meaning
            best = d
    return best_idx, best_meaning

def handle_special_char(text): #handling #
    newtext = []
    for token in text.split():
        if "#" in token:
            token = token.split("#")[0]
        newtext.append(token)
    return " ".join(newtext)

def count_intersections(s1, s2):
    tokens1, tokens2 = set(s1.split()), set(s2.split())
    return len(tokens1.intersection(tokens2))