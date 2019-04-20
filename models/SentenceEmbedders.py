import re
from nltk.tokenize import RegexpTokenizer

import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm
"""
class ELMo():
    def __init__(self, elmo):
        self.elmo = elmo
"""


def ELMoMultipleChoiceEmbedder(elmo,
                               dataframe=None,
                               batch_size=64,
                               tokenizer=None,
                               pooling=None,
                               **kwargs):
    try:
        num_samples = len(dataframe)
    except Exception:
        try:
            num_samples = len(kwargs['questions'])
        except Exception:
            raise TypeError('Questions are provided in unsupported format')

    if not tokenizer:
        tokenizer = RegexpTokenizer('[А-Яа-яёA-Za-z0-9]+')
    if dataframe is not None:
        question_col = kwargs['question_col']
        answer_choices_cols = kwargs['answer_choices_cols']
        correct_answer_col = kwargs['correct_answer_col']

        question_data = dataframe[question_col].values
        answer_choices_data = {}
        for i in range(len(answer_choices_cols)):
            answer_choices_data[f'choice_{i}'] = dataframe[
                answer_choices_cols[i]].values
        correct_answers_data = dataframe[correct_answer_col]

    else:
        question_data = kwargs['questions']
        num_choices = len(kwargs['answer_choices'])
        answer_choices_data = {}
        for i in range(len(num_choices)):
            if isinstance(kwargs['answer_choices'], dict):
                answer_choices_data[f'choice_{i}'] = kwargs['answer_choices'][
                    f'choice_{i}']
            elif isinstance(kwargs['answer_choices'], list):
                answer_choices_data[f'choice_{i}'] = kwargs['answer_choices'][
                    i]
        correct_answers_data = kwargs['correct_answers']

    embeddings = []
    for batch in tqdm(range(0, num_samples - batch_size, batch_size)):
        ids = []
        questions = []
        answer_choices = {choice: [] for choice in answer_choices_data}
        correct_answers = []

        for i in range(batch, batch + batch_size):
            ids.append(i)
            questions.append(tokenizer.tokenize(question_data[i]))
            for choice in answer_choices:
                answer_choices[choice].append(
                    tokenizer.tokenize(answer_choices_data[choice][i]))
            correct_answers.append(correct_answers_data[i])

        questions_elmo_batch = elmo(questions)
        choices_elmo_batch = {
            choice: elmo(answer_choices[choice])
            for choice in answer_choices
        }
        for j in range(len(questions_elmo_batch)):
            if pooling == 'mean':
                sample = {'question': np.mean(questions_elmo_batch[j], axis=0)}
                for choice in choices_elmo_batch:
                    sample[choice] = np.mean(
                        choices_elmo_batch[choice][j], axis=0)
            elif pooling == 'max':
                sample = {'question': np.max(questions_elmo_batch[j], axis=0)}
                for choice in choices_elmo_batch:
                    sample[choice] = np.max(
                        choices_elmo_batch[choice][j], axis=0)
            elif pooling == 'mean_max':
                sample = {
                    'question':
                    np.concatenate([
                        np.mean(questions_elmo_batch[j], axis=0),
                        np.max(questions_elmo_batch[j], axis=0)
                    ])
                }
                for choice in choices_elmo_batch:
                    sample[choice] = np.concatenate([
                        np.mean(choices_elmo_batch[choice][j], axis=0),
                        np.max(choices_elmo_batch[choice][j], axis=0)
                    ])
            else:
                sample = {'question': questions_elmo_batch[j]}
                for choice in choices_elmo_batch:
                    sample[choice] = choices_elmo_batch[choice][j]
            sample['id'] = ids[j]
            sample['correct_answer'] = correct_answers[j]
            embeddings.append(sample)
    return embeddings


def ELMoParaphraseEmbedder(elmo,
                           dataframe=None,
                           batch_size=64,
                           tokenizer=None,
                           pooling=None,
                           **kwargs):
    try:
        num_samples = len(dataframe)
    except Exception:
        try:
            num_samples = len(kwargs['text_1'])
        except Exception:
            raise TypeError('Questions are provided in unsupported format')

    if not tokenizer:
        tokenizer = RegexpTokenizer('[А-Яа-яёA-Za-z0-9]+')
    if dataframe is not None:
        text_1_col = kwargs['text_1_col']
        text_2_col = kwargs['text_2_col']
        correct_answer_col = kwargs['correct_answer_col']

        text_1_data = dataframe[text_1_col].values
        text_2_data = dataframe[text_2_col].values
        correct_answers_data = dataframe[correct_answer_col]

    else:
        text_1_data = kwargs['text_1']
        text_2_data = kwargs['text_2']
        correct_answers_data = kwargs['correct_answers']

    embeddings = []
    for batch in tqdm(range(0, num_samples - batch_size, batch_size)):
        ids = []
        text_1 = []
        text_2 = []
        correct_answers = []

        for i in range(batch, batch + batch_size):
            ids.append(i)
            text_1.append(tokenizer.tokenize(text_1_data[i]))
            text_2.append(tokenizer.tokenize(text_2_data[i]))
            correct_answers.append(correct_answers_data[i])

        text_1_elmo_batch = elmo(text_1)
        text_2_elmo_batch = elmo(text_2)
        for j in range(len(text_1_elmo_batch)):
            if pooling == 'mean':
                sample = {
                    'text_1': np.mean(text_1_elmo_batch[j], axis=0),
                    'text_2': np.mean(text_2_elmo_batch[j], axis=0)
                }
            elif pooling == 'max':
                sample = {
                    'text_1': np.max(text_1_elmo_batch[j], axis=0),
                    'text_2': np.max(text_2_elmo_batch[j], axis=0)
                }
            elif pooling == 'mean_max':
                sample = {
                    'text_1':
                    np.concatenate([
                        np.mean(text_1_elmo_batch[j], axis=0),
                        np.max(text_1_elmo_batch[j], axis=0)
                    ]),
                    'text_2':
                    np.concatenate([
                        np.mean(text_2_elmo_batch[j], axis=0),
                        np.max(text_2_elmo_batch[j], axis=0)
                    ])
                }
            else:
                sample = {
                    'text_1': text_1_elmo_batch[j],
                    'text_2': text_2_elmo_batch[j]
                }
            sample['id'] = ids[j]
            sample['correct_answer'] = correct_answers[j]
            embeddings.append(sample)
    return embeddings


def BertMultipleChoiceEmbedder(bert_encoder,
                               dataframe=None,
                               batch_size=64,
                               mean=False,
                               **kwargs):
    try:
        num_samples = len(dataframe)
    except Exception:
        try:
            num_samples = len(kwargs['questions'])
        except Exception:
            raise TypeError('Questions are provided in unsupported format')

    if dataframe is not None:
        question_col = kwargs['question_col']
        answer_choices_cols = kwargs['answer_choices_cols']
        correct_answer_col = kwargs['correct_answer_col']

        question_data = dataframe[question_col].values
        answer_choices_data = {}
        for i in range(len(answer_choices_cols)):
            answer_choices_data[f'choice_{i}'] = dataframe[
                answer_choices_cols[i]].values
        correct_answers_data = dataframe[correct_answer_col]

        answer_choices = {}
        for i in range(len(answer_choices_cols)):
            answer_choices[f'choice_{i}'] = dataframe[answer_choices_cols[i]]
    else:
        question_data = kwargs['questions']
        num_choices = len(kwargs['answer_choices'])
        answer_choices_data = {}
        for i in range(len(num_choices)):
            if isinstance(kwargs['answer_choices'], dict):
                answer_choices_data[f'choice_{i}'] = kwargs['answer_choices'][
                    f'choice_{i}']
            elif isinstance(kwargs['answer_choices'], list):
                answer_choices_data[f'choice_{i}'] = kwargs['answer_choices'][
                    i]
        correct_answers_data = kwargs['correct_answers']

    embeddings = []
    for batch in tqdm(range(0, num_samples - batch_size, batch_size)):
        ids = []
        questions = []
        answer_choices = {choice: [] for choice in answer_choices_data}
        correct_answers = []

        for i in range(batch, batch + batch_size):
            ids.append(i)
            questions.append(question_data[i])
            for choice in answer_choices:
                answer_choices[choice].append(answer_choices_data[choice][i])
            correct_answers.append(correct_answers_data[i])

        questions_bert_batch = bert_encoder(questions)
        choices_bert_batch = {
            choice: bert_encoder(answer_choices[choice])
            for choice in answer_choices
        }
        for j in range(len(questions_bert_batch)):
            sample = {'question': questions_bert_batch[j]}
            for choice in choices_bert_batch:
                sample[choice] = choices_bert_batch[choice][j]
            sample['id'] = ids[j]
            sample['correct_answer'] = correct_answers[j]
            embeddings.append(sample)
    return embeddings


def BertParaphraseEmbedder(bert_encoder,
                           dataframe=None,
                           batch_size=64,
                           tokenizer=None,
                           mean=False,
                           **kwargs):
    try:
        num_samples = len(dataframe)
    except Exception:
        try:
            num_samples = len(kwargs['text_1'])
        except Exception:
            raise TypeError('Questions are provided in unsupported format')

    if not tokenizer:
        tokenizer = RegexpTokenizer('[А-Яа-яёA-Za-z0-9]+')
    if dataframe is not None:
        text_1_col = kwargs['text_1_col']
        text_2_col = kwargs['text_2_col']
        correct_answer_col = kwargs['correct_answer_col']

        text_1_data = dataframe[text_1_col].values
        text_2_data = dataframe[text_2_col].values
        correct_answers_data = dataframe[correct_answer_col]

    else:
        text_1_data = kwargs['text_1']
        text_2_data = kwargs['text_2']
        correct_answers_data = kwargs['correct_answers']

    embeddings = []
    for batch in tqdm(range(0, num_samples - batch_size, batch_size)):
        ids = []
        text_1 = []
        text_2 = []
        correct_answers = []

        for i in range(batch, batch + batch_size):
            ids.append(i)
            text_1.append(text_1_data[i])
            text_2.append(text_2_data[i])
            correct_answers.append(correct_answers_data[i])

        text_1_elmo_batch = bert_encoder(text_1)
        text_2_elmo_batch = bert_encoder(text_2)
        for j in range(len(text_1_elmo_batch)):
            if mean:
                sample = {
                    'text_1': np.mean(text_1_elmo_batch[j], axis=0),
                    'text_2': np.mean(text_2_elmo_batch[j], axis=0)
                }
            else:
                sample = {
                    'text_1': text_1_elmo_batch[j],
                    'text_2': text_2_elmo_batch[j]
                }
            sample['id'] = ids[j]
            sample['correct_answer'] = correct_answers[j]
            embeddings.append(sample)
    return embeddings


def max_pool(arr):
    arr = arr.reshape(-1, 768)
    return np.max(arr, axis=0)


def avg_pool(arr):
    arr = arr.reshape(-1, 768)
    return np.mean(arr, axis=0)


def bert_layer(embs,
               layer_num=12,
               strategy='concat',
               problem_type='multiple-choice'):

    embs_layer = []
    for emb in embs:
        if strategy == 'concat':
            pooling = lambda x: x
        elif strategy == 'max':
            pooling = max_pool
        elif strategy == 'mean':
            pooling = avg_pool
        elif strategy == 'mean_max':
            pooling = lambda x: np.concatenate([avg_pool(x), max_pool(x)])

        if isinstance(layer_num, int):
            layer_num = (layer_num, layer_num)
            pooling = lambda x: x
        if problem_type == 'multiple-choice':
            sample = {
                'question':
                pooling(emb['question'][768 * (layer_num[0] - 1):768 *
                                        (layer_num[1])])
            }
            for i in range(4):
                sample[f'choice_{i}'] = pooling(
                    emb[f'choice_{i}'][768 * (layer_num[0] - 1):768 *
                                       (layer_num[1])])
        elif problem_type == 'paraphrase':
            sample = {
                'text_1':
                pooling(emb['text_1'][768 * (layer_num[0] - 1):768 *
                                      (layer_num[1])]),
                'text_2':
                pooling(emb['text_2'][768 * (layer_num[0] - 1):768 *
                                      (layer_num[1])])
            }
        sample['id'] = emb['id']
        sample['correct_answer'] = emb['correct_answer']
        embs_layer.append(sample)
    return embs_layer
