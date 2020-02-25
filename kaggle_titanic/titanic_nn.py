import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

    # -------------------------------------------
    # Data input
    df = pd.read_csv('../data/train.csv')
    # print(df.to_string())

    # -------------------------------------------
    # Exploratory data analysis

    # --------------------------------
    # What is the base survival rate?
    # 38%
    # total = df.shape[0]
    # survived = df[df['Survived'] == 1].shape[0]
    # print(survived/total)

    # -------------------------------------------
    # Training
    df_train, df_validate, df_test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    features_col = ["Fare", "Pclass", "SibSp", "Parch"]

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    print(df_train[features_col].isna().sum())
    clf.fit(df_train[features_col],
            df_train["Survived"])

    # -------------------------------------------
    # Testing
    predictions = clf.predict(df_validate[features_col])
    df_validate["predicted"] = predictions
    print(accuracy_score(df_validate["Survived"], predictions))
