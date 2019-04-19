import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import accuracy_score, classification_report, f1_score
from .SentenceEmbedders import bert_layer
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


class ExperimentResult():
    def __init__(self, log=['embeddings_model', 'layers', 'pooling', 'accuracy', 'f1']):
        self.log = {name: [] for name in log}
    def append(self, **kwargs):
        for name in self.log:
            self.log[name].append(kwargs[name])
    def show_results(self):
        return pd.DataFrame.from_dict(self.log)
    
def run_similarity_experiments(experiment_log,
                               embeddings, 
                               embeddings_model='bert-as-service', 
                               problem_type='multiple-choice',
                               strategy=None, single_layers=None, layers=None, 
                               validation_size=0.3, num_thresholds=50):
    if (embeddings_model == 'bert-as-service' and problem_type == 'multiple-choice'):
        experiments = ExperimentResult()
        for l in tqdm(single_layers, desc='Single layers'):
            layer = bert_layer(embeddings, layer_num=l)
            y, y_pred = mc_similarity_predict(layer)
            experiment_log.append(embeddings_model='bert-as-service', 
                           layers=l, pooling='single_layer', 
                           accuracy=round(f1_score(y, y_pred, average='micro'), 3), 
                           f1=round(f1_score(y, y_pred, average='macro'), 3))

        for s in tqdm(strategy, desc='Strategies'):
            for l in layers:
                layer = bert_layer(embeddings, layer_num=l, strategy=s)
                y, y_pred = mc_similarity_predict(layer)
                experiment_log.append(embeddings_model='bert-as-service', 
                           layers=l, pooling=s, 
                           accuracy=round(f1_score(y, y_pred, average='micro'), 3), 
                           f1=round(f1_score(y, y_pred, average='macro'), 3))
        return experiment_log.show_results()
    
    if (embeddings_model == 'bert-as-service' and problem_type == 'paraphrase'):
        experiments = ExperimentResult()
        for l in tqdm(single_layers, desc='Single layers'):
            layer = bert_layer(embeddings, layer_num=l, problem_type='paraphrase')
            acc, f1 = paraphrase_similarity_predict(layer, 
                                                    num_thresholds=num_thresholds, 
                                                    validation_size=validation_size)
            experiment_log.append(embeddings_model='bert-as-service', 
                           layers=l, pooling='single_layer', 
                           accuracy=round(acc, 3), 
                           f1=round(f1, 3))

        for s in tqdm(strategy, desc='Strategies'):
            for l in layers:
                layer = bert_layer(embeddings, layer_num=l, strategy=s, problem_type='paraphrase')
                acc, f1 = paraphrase_similarity_predict(layer, 
                                                        num_thresholds=num_thresholds, 
                                                        validation_size=validation_size)
                experiment_log.append(embeddings_model='bert-as-service', 
                           layers=l, pooling=s, 
                           accuracy=round(acc, 3), 
                           f1=round(f1, 3))
        return experiment_log.show_results()

# -------------------------------------------------------------------------------------------------

def mc_similarity_predict(embeddings, topk=1, distance='cosine_distance'):
    """
    Return true labels and predicted labels. If topk > 1 then the function
    will return topk number of closest answers to question
    
    embeddings:
    type: a list of dictionaries, where each dictionary contains 
          correct answer and embeddings for question and answers
    ------------------------------------------------------------
    
    topk: number of closest answers to return
    ------------------------------------------------------------
    
    distance: type of distance
    avaliable values: 'cosine_distance', 'cosine_similarity'
    """
    y_pred, y = [], []
    for i in range(len(embeddings)):
        # prediction
        distances = np.zeros(4)
        for j in range(0, 4):
            if distance == 'cosine_similarity':    
                distances[j] = cosine_similarity(embeddings[i]['question'].reshape(1, -1), 
                                                 embeddings[i][f'choice_{j}'].reshape(1, -1))[0][0]
            elif distance == 'cosine_distance':
                distances[j] = cosine_distances(embeddings[i]['question'].reshape(1, -1), 
                                                 embeddings[i][f'choice_{j}'].reshape(1, -1))[0][0]
        if distance == 'cosine_similarity':
            if topk == 1:
                y_pred.append(np.argmax(distances))
            else:
                y_pred.append(np.argsort(distances)[::-1][:topk])
        elif distance == 'cosine_distance':
            if topk == 1:
                y_pred.append(np.argmin(distances))
            else:
                y_pred.append(np.argsort(distances)[:topk])
        else:
            raise ValueError(f'{distance} is not supported')
        
        
        # true labels
        y.append(embeddings[i]['correct_answer'])
    return y, y_pred



def topk_accuracy(y, y_pred, k=2):
    """
    Returns ration of cases in which the correct answer 
    is in k closest
    """
    pred = []
    for i in range(len(y)):
        if y[i] in y_pred[i][:k]:
            pred.append(1)
        else:
            pred.append(0)
    return sum(pred) / len(pred)


from sklearn.metrics import accuracy_score, classification_report, f1_score
def paraphrase_similarity_predict(embeddings, num_thresholds=10, 
                                  threshold=None, validation_size=None, 
                                  random_seed=42):
    """
    
    """
    y = []
    distances = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        distances[i] = cosine_distances(embeddings[i]['text_1'].reshape(1, -1), 
                                        embeddings[i]['text_2'].reshape(1, -1))[0][0]
        y.append(embeddings[i]['correct_answer'])
    
    if threshold:
        y_pred = np.zeros(len(y))
        y_pred[np.where(distances < threshold)[0]] = 1
        return y, y_pred
    
    if validation_size:
        np.random.seed(random_seed)
        n = len(embeddings)
        idx = np.random.choice(n, n, replace=False)
        split = int(n * (1 - validation_size))
        x_train, y_train = distances[idx[:split]], np.array(y)[idx[:split]]
        x_test, y_test = distances[idx[split:]], np.array(y)[idx[split:]]
        
        min_dist, max_dist = min(x_train), max(x_train)
        xrange = np.linspace(min_dist, max_dist, num_thresholds + 2)[1:-1]
        optimum_acc = []
        optimum_f1 = []
        for t in xrange:
            y_pred = np.zeros(len(y_train))
            y_pred[np.where(x_train < t)[0]] = 1
            acc = accuracy_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            optimum_acc.append((acc, t))
            optimum_f1.append((f1, t))
        opt_acc_t = max(optimum_acc)[1]
        opt_f1_t = max(optimum_f1)[1]
        
        y_pred_acc = np.zeros(len(y_test))
        y_pred_acc[np.where(x_test < opt_acc_t)[0]] = 1
        
        y_pred_f1 = np.zeros(len(y_test))
        y_pred_f1[np.where(x_test < opt_f1_t)[0]] = 1
        
        final_acc = (accuracy_score(y_test, y_pred_acc), f1_score(y_test, y_pred_acc))
        final_f1 = (accuracy_score(y_test, y_pred_f1), f1_score(y_test, y_pred_f1))
        return max(final_acc[0], final_f1[0]), max(final_acc[1], final_f1[1])
    
    
    min_dist, max_dist = min(distances), max(distances)
    xrange = np.linspace(min_dist, max_dist, num_thresholds + 2)[1:-1]
    optimum_acc = []
    optimum_f1 = []
    for t in xrange:
        y_pred = np.zeros(len(y))
        y_pred[np.where(distances < t)[0]] = 1
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        optimum_acc.append((acc, t))
        optimum_f1.append((f1, t))
    return max(optimum_acc), max(optimum_f1)


# -------------------------------------------------------------------------------------------------

def run_inputs_experiments(experiment_log,
                           embeddings, 
                           classifier='CatBoost',
                           classifier_params=None,
                           embeddings_model='bert-as-service', 
                           problem_type='multiple-choice',
                           strategy=None, single_layers=None, layers=None, 
                           validation_size=0.3, num_thresholds=50):
    
    if (embeddings_model == 'bert-as-service' and problem_type == 'multiple-choice'):
        experiments = ExperimentResult()
        for l in tqdm(single_layers, desc='Single layers'):
            if classifier == 'CatBoost':
                if classifier_params:
                    model = CatBoostClassifier(**classifier_params)
                else:
                    model = CatBoostClassifier(iterations=100, 
                                               task_type = 'GPU')
            elif classifier == 'LogReg':
                if classifier_params:
                    model = LogisticRegression(**classifier_params)
                else:
                    model = LogisticRegression(random_state=42)
            
            layer = bert_layer(embeddings, layer_num=l)
            acc, f1 = mc_inputs_predict(layer, model)
            experiment_log.append(embeddings_model='bert-as-service', 
                                  classifier=classifier,
                                  layers=l, pooling='single_layer', 
                                  accuracy=round(acc, 3), 
                                  f1=round(f1, 3))

        for s in tqdm(strategy, desc='Strategies'):
            for l in layers:
                if classifier == 'CatBoost':
                    if classifier_params:
                        model = CatBoostClassifier(**classifier_params)
                    else:
                        model = CatBoostClassifier(iterations=100, 
                                                   task_type = 'GPU')
                elif classifier == 'LogReg':
                    if classifier_params:
                        model = LogisticRegression(**classifier_params)
                    else:
                        model = LogisticRegression(random_state=42)
                
                layer = bert_layer(embeddings, layer_num=l, strategy=s)
                acc, f1 = mc_inputs_predict(layer, model)
                experiment_log.append(embeddings_model='bert-as-service',
                                      classifier=classifier,
                                      layers=l, pooling=s, 
                                      accuracy=round(acc, 3), 
                                      f1=round(f1, 3))
        return experiment_log.show_results()
    
    if (embeddings_model == 'bert-as-service' and problem_type == 'paraphrase'):
        experiments = ExperimentResult()
        for l in tqdm(single_layers, desc='Single layers'):
            if classifier == 'CatBoost':
                if classifier_params:
                    model = CatBoostClassifier(**classifier_params)
                else:
                    model = CatBoostClassifier(iterations=100, 
                                               task_type = 'GPU')
            elif classifier == 'LogReg':
                if classifier_params:
                    model = LogisticRegression(**classifier_params)
                else:
                    model = LogisticRegression(random_state=42)
            
            layer = bert_layer(embeddings, layer_num=l, problem_type='paraphrase')
            acc, f1 = paraphrase_inputs_predict(layer, model)
            experiment_log.append(embeddings_model='bert-as-service', 
                                  classifier=classifier,
                                  layers=l, pooling='single_layer', 
                                  accuracy=round(acc, 3), 
                                  f1=round(f1, 3))

        for s in tqdm(strategy, desc='Strategies'):
            for l in layers:
                if classifier == 'CatBoost':
                    if classifier_params:
                        model = CatBoostClassifier(**classifier_params)
                    else:
                        model = CatBoostClassifier(iterations=100, 
                                                   task_type = 'GPU')
                elif classifier == 'LogReg':
                    if classifier_params:
                        model = LogisticRegression(**classifier_params)
                    else:
                        model = LogisticRegression(random_state=42)
                
                layer = bert_layer(embeddings, layer_num=l, strategy=s, problem_type='paraphrase')
                acc, f1 = paraphrase_inputs_predict(layer, model)
                experiment_log.append(embeddings_model='bert-as-service',
                                      classifier=classifier,
                                      layers=l, pooling=s, 
                                      accuracy=round(acc, 3), 
                                      f1=round(f1, 3))
        return experiment_log.show_results()



def prepare_for_supervised(embeddings):
    """
    Returns the list of the concatenated pairs of question
    and answers and the list of target labels
    
    embeddigns: 
    type: a list of dictionaries, where each dictionary contains 
          correct answer and embeddings for question and answers
    """
    
    X, y = [], []
    for problem in embeddings:
        pairs = [np.concatenate((problem['question'], problem[i])) 
                 for i in ('choice_0', 'choice_1', 'choice_2', 'choice_3')]
        correct_answer = problem['correct_answer']
        
        labels = [0, 0, 0, 0]
        labels[correct_answer] = 1
        X.extend(pairs)
        y.extend(labels)
    return X, y


def make_pairs(dct):
    pairs = []
    for i in range(0, 4):
        pairs.append(np.concatenate((dct['question'], dct[f'choice_{i}'])))
    return pairs

# prediction of the answer to the given question
def predict_probs(embeddings, model, topk=1, predict_proba=True):
    pred = []
    for problem in embeddings:
        pairs = make_pairs(problem)
        if predict_proba:
            probs = model.predict_proba(pairs)[:, 1]
        else:
            probs = model.decision_function(pairs)
        if topk == 1:
            pred.append(np.argmax(probs))
        else:
            pred.append(np.argsort(probs)[::-1][:topk])
    return pred


def mc_inputs_predict(embeddings,
                      classifier,
                      problem_type='multiple-choice', 
                      validation_size=0.25,
                      random_seed=42):
    np.random.seed(random_seed)
    n = len(embeddings)
    idx = np.random.choice(n, n, replace=False)
    split = int(n * (1 - validation_size))
    
    y = np.array([embs['correct_answer'] for embs in embeddings])
    train_samples = np.array(embeddings)[idx[:split]]
    test_samples, y_test = np.array(embeddings)[idx[split:]], y[idx[split:]]
    
    X_train, y_train = prepare_for_supervised(train_samples)
    classifier.fit(X_train, y_train)
    
    pred = predict_probs(test_samples, classifier)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro')

def paraphrase_inputs_predict(embeddings,
                              classifier,
                              validation_size=0.25,
                              random_seed=42):
    np.random.seed(random_seed)
    n = len(embeddings)
    idx = np.random.choice(n, n, replace=False)
    split = int(n * (1 - validation_size))
    
    y = np.array([embs['correct_answer'] for embs in embeddings])
    samples = np.array([np.concatenate([embs['text_1'], embs['text_2']]) for embs in embeddings])
    train_samples, y_train = samples[idx[:split]], y[idx[:split]]
    test_samples, y_test = samples[idx[split:]], y[idx[split:]]
    
    
    classifier.fit(train_samples, y_train)
    
    pred = classifier.predict(test_samples)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro')
