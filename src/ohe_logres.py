import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load the full training data with the folds

    df = pd.read_csv('../input/cat_data/cat_train_folds.csv')

    # all the columns would be used as features for this model
    # except the id, kfolds, and the target columns

    # use list of lists to extract all the columns you need and
    # store it in a features variable

    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]

    # fill all NaN values with NONE
    # it should be noted that all the columns are converted to string but it doesn't
    # matter because all the features in this dataset are categorical i.e finite.

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data still using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize one hot encoder from sklearn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    ohe.fit(full_data)

    # trasnform training data
    x_train = ohe.transform(df_train[features])

    x_valid = ohe.transform(df_valid[features])

    # initialize logistic regression model
    model = linear_model.LogisticRegression()

    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f'FOld = {fold}, AUC= {auc}')


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)