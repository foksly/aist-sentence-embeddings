import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# save and load functions
def save2pkl(path, file):
    """
    Saves to pkl
    """
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(file, f)


def load_pkl(path):
    """
    Loads pkl file
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# this function does not work correctly with out of vocabulary trigrams
def fasttext_embeddings(df, model):
    embs = []
    prog = re.compile('[А-Яа-яёA-Za-z0-9]+')
    num = re.compile('[0-4]+')
    for i in tqdm(range(len(df))):
        question = np.mean(model.wv[prog.findall(df.iloc[i]['q'])], axis=0)
        answer1 = np.mean(model.wv[prog.findall(df.iloc[i]['a1'])], axis=0)
        answer2 = np.mean(model.wv[prog.findall(df.iloc[i]['a2'])], axis=0)
        answer3 = np.mean(model.wv[prog.findall(df.iloc[i]['a3'])], axis=0)
        answer4 = np.mean(model.wv[prog.findall(df.iloc[i]['a4'])], axis=0)
        if isinstance(df.iloc[i]['ca'], int):
            correct_answer = df.iloc[i]['ca'] - 1
        else:
            correct_answer = int(num.findall(df.iloc[i]['ca'])[0])
            if correct_answer != 0:
                correct_answer -= 1

        embs.append({
            'q': question,
            'a1': answer1,
            'a2': answer2,
            'a3': answer3,
            'a4': answer4,
            'ca': correct_answer
        })
    return embs
