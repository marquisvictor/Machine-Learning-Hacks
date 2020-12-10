# src/train.py
import os
import config
import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import model_dispatcher


def run(fold, model):
    # read the training data to pandas with the folds

    df = pd.read_csv(config.TRAINING_FILE)
    # note that the training data is where kfold is not equal to provided fold,
    # also note that we reset the index.
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from the train_df and convert it to a numpy array
    # using `.values`.
    # target is denoted as label column in the dataframe

    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # similarly for validation, we have
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    # initialize simple decision tree classifierw
    clf = model_dispatcher.models[model]

    # fit the model on the training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold={fold}, Accuracy={accuracy}')

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f'dt_{fold}.bin'))


if __name__ == '__main__':
    # initializer the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # now you should add the different arguments you need and their type
    # currently, we need only the fold argument to specify how many
    # folds we want to train the machine learning model for.

    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    run(fold=args.fold, model=args.model)

    # run(fold=0)
    # run(fold=1)
    # run(fold=2)
    # run(fold=3)
    # run(fold=4)