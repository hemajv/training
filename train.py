import os
import yaml
import pprint
import mlflow

import numpy as np
import dask.array as da
from dask_ml.metrics import log_loss

from rgf.sklearn import RGFClassifier
from sklearn.metrics import precision_recall_fscore_support


def get_hyperparameters(job_id, hyperparams_fname="hyperparameters.yml"):
    # Update file name with correct path
    with open(hyperparams_fname, 'r') as stream:
        hyperparam_set = yaml.load(stream)

    print("\nHypermeter set for job_id: ", job_id)
    print("------------------------------------")
    pprint.pprint(hyperparam_set[job_id]["hyperparam_set"])
    print("------------------------------------\n")

    return hyperparam_set[job_id]["hyperparam_set"]


def calculate_value(ones, tens, hundreds):
    # log hyperparams for this run
    mlflow.log_param('ones', ones)
    mlflow.log_param('tens', tens)
    mlflow.log_param('hundreds', hundreds)

    # calculate value
    result = ones + 10*tens + 100*hundreds

    # assume ground truth is whether result was correct or not
    truth = int(str(hundreds) + str(tens) + str(ones))
    score = truth==result

    # how well did the model do this run
    mlflow.log_metric('is_correct', score)


def train(params):
    # log hyperparams for this run
    for k,v in params.items():
        mlflow.log_param(k, v)

    # load dataset files
    dataset = np.load('preprocessed/dataset.npz')
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    Y_train = dataset['Y_train']
    Y_test = dataset['Y_test']

    # instantiate model with params
    rgf_clf = RGFClassifier(**params)
    rgf_clf.fit(X_train, Y_train)

    # predict on test data
    Y_pred = rgf_clf.predict(X_test)
    Y_pred_proba = rgf_clf.predict_proba(X_test)

    # log logistic loss value
    logistic_loss = log_loss(Y_test, Y_pred_proba)
    mlflow.log_metric('log_loss', logistic_loss)

    # log precision, recall, f1
    p, r, f, _ = precision_recall_fscore_support(y_true=Y_test, y_pred=Y_pred, average='binary')
    mlflow.log_metric('precision', p)
    mlflow.log_metric('recall', r)
    mlflow.log_metric('f1', f)


if __name__ == "__main__":
    # get the id that will be used to access a hyperparam set
    job_id = os.environ.get("HYPERPARAM_SET_ID")
    if job_id is None:
        raise EnvironmentError("Could not find variable HYPERPARAM_SET_ID in environment. It must be defined for job to run")

    # openshift pods complaint this being not an int
    job_id = int(job_id)

    # get hyperparameters for this specific job
    currjob_hyperparams = get_hyperparameters(job_id)

    # run training
    train(currjob_hyperparams)
