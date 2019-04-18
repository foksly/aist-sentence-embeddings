import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# remove hard questions
def add_isYear(df):
    """
    Adds isYear column in which 1 corresponds to questions
    where answers are dates, otherwise 0

    df: type DataFrame
    """
    isYear = []
    for i in range(len(df)):
        match = re.match(r'.*([1-3][0-9]{3})', df['a1'].iloc[i])
        if match is not None:
            isYear.append(1)
        else:
            isYear.append(0)
    df['isYear'] = isYear


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


# build embeddings functions
def elmo_embeddings(df,
                    elmo,
                    context_col='q',
                    answers_cols=['a1', 'a2', 'a3', 'a4'],
                    ca_col='ca',
                    from0to3=False):
    """
    Returns list of dictionaries. 
    Each dictionary has the following format:
    {'id': id of question in DataFrame,
     'q': <question embedding>, 
     'a_i': <i-th answer embedding>,
     'ca': correct answer}
     
    df: DataFrame with questions
    format: columns = ['q', 'a1, ..., 'a4', 'ca']
    ---------------------------------------------
    
    elmo: DeepPavlov elmo embedder
    """
    embeddings = []
    prog = re.compile('[А-Яа-яёA-Za-z0-9]+')
    num = re.compile('[0-4]+')
    for i in tqdm(range(len(df))):
        problems = {}
        q = prog.findall(df.iloc[i][context_col])
        a1 = prog.findall(df.iloc[i][answers_cols[0]])
        a2 = prog.findall(df.iloc[i][answers_cols[1]])
        a3 = prog.findall(df.iloc[i][answers_cols[2]])
        a4 = prog.findall(df.iloc[i][answers_cols[3]])
        if isinstance(df.iloc[i][ca_col], int):
            if not from0to3:
                correct_answer = df.iloc[i][ca_col] - 1
            else:
                correct_answer = df.iloc[i][ca_col]
        else:
            if isinstance(df.iloc[i][ca_col], str):
                correct_answer = int(num.findall(df.iloc[i][ca_col])[0])
            else:
                try:
                    correct_answer = int(df.iloc[i][ca_col])
                except Exception:
                    raise TypeError
            if correct_answer != 0:
                if not from0to3:
                    correct_answer -= 1

        embs = elmo([q, a1, a2, a3, a4])

        problems = {
            'id': i,
            'q': embs[0],
            'a1': embs[1],
            'a2': embs[2],
            'a3': embs[3],
            'a4': embs[4],
            'ca': correct_answer
        }
        embeddings.append(problems)
    return embeddings


def elmo_batch_embedder(df,
                        elmo,
                        context_col='q',
                        answers_cols=['a1', 'a2', 'a3', 'a4'],
                        ca_col='ca',
                        from0to3=False,
                        batch_size=64):
    embeddings = []
    for j in tqdm(range(0, len(df) - batch_size, batch_size)):
        prog = re.compile('[А-Яа-яёA-Za-z0-9]+')
        num = re.compile('[0-4]+')
        ids = []
        questions = []
        answers1 = []
        answers2 = []
        answers3 = []
        answers4 = []
        correct_answers = []
        for i in tqdm(range(j, j + batch_size)):
            problems = {}
            q = prog.findall(df.iloc[i][context_col])
            a1 = prog.findall(df.iloc[i][answers_cols[0]])
            a2 = prog.findall(df.iloc[i][answers_cols[1]])
            a3 = prog.findall(df.iloc[i][answers_cols[2]])
            a4 = prog.findall(df.iloc[i][answers_cols[3]])
            if isinstance(df.iloc[i][ca_col], int):
                if not from0to3:
                    correct_answer = df.iloc[i][ca_col] - 1
                else:
                    correct_answer = df.iloc[i][ca_col]
            else:
                if isinstance(df.iloc[i][ca_col], str):
                    correct_answer = int(num.findall(df.iloc[i][ca_col])[0])
                else:
                    try:
                        correct_answer = int(df.iloc[i][ca_col])
                    except Exception:
                        raise TypeError
                if correct_answer != 0:
                    if not from0to3:
                        correct_answer -= 1
            ids.append(i)
            questions.append(q)
            answers1.append(a1)
            answers2.append(a2)
            answers3.append(a3)
            answers4.append(a4)
            correct_answers.append(correct_answer)
        qustions_elmo = elmo(questions)
        answers1_elmo = elmo(answers1)
        answers2_elmo = elmo(answers2)
        answers3_elmo = elmo(answers3)
        answers4_elmo = elmo(answers4)
        embeddings.extend([{
            'id': ids[k],
            'q': qustions_elmo[k],
            'a1': answers1_elmo[k],
            'a2': answers2_elmo[k],
            'a3': answers3_elmo[k],
            'a4': answers4_elmo[k],
            'ca': correct_answers[k]
        }] for k in range(len(questions_elmo)))
        # embeddings.append(problems)
    return embeddings


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
