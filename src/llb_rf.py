import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load the full training data with folds

    df = pd.read_csv('../input/cat_data/cat_train_folds.csv')

    # select all columns we need and store in a list array, except
    # the id, target, and kfold columns

    features = [f for f in df.columns if f not in ('target', 'id', 'kfold')]

    # fill all the NaN values with NONE

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # NOW ITS TIME TO LABEL ENCODE THE FEATURES

    for col in features:
        # initialize label encoder for each feature column

        lbl = preprocessing.LabelEncoder()

        # fit the label encoder on the full data

        lbl.fit(df[col])

        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize random forest model

    model = ensemble.RandomForestClassifier()

    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    #print auc

    print(f'Fold = {fold}, AUC = {auc}')


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
