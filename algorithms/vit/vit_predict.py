import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import make_pipeline


# carrega os datasets
def main(df):
    train = pd.read_csv("X_train.csv")
    test = pd.read_csv("X_test.csv")
    ytest = pd.read_csv("y_test.csv", names=["target"])
    return df

# normaliza entre 0 e 1
x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)

# verifica valores vazios/nulos e exclui
train.isnull()
train.dropna(inplace=True)

# faz clusterização
kmeans = KMeans(n_clusters=2)
kmeans.fit(train)

target = kmeans.labels_

df = pd.DataFrame(kmeans.labels_)
df.columns = ['target']

X = train
y = df

# junta os dados de treino e o target
df2 = pd.concat([train, df], axis=1)

# remove os dados faltantes
df2.dropna(inplace=True)

X = df2
y = df2['target']

kf = KFold(n_splits=2, random_state=0, shuffle=True)

second_level = np.zeros((X.shape[0], 4))

for tr, ts in kf.split(X, y):
    Xtr, Xval = X.iloc[tr], X.iloc[ts]
    ytr, yval = y.iloc[tr], y.iloc[ts]

    rf = RandomForestClassifier(n_estimators=100, n_jobs=6, random_state=10)
    rf.fit(Xtr, ytr)
    prf = rf.predict_proba(Xval)[:, 1]
    prf_ = (prf > 0.5).astype(int)

    print("RF Accuracy: {} - Log Loss: {}".format(accuracy_score(yval, prf_), log_loss(yval, prf)))

    et = ExtraTreesClassifier(n_estimators=100, n_jobs=6, random_state=10)
    et.fit(Xtr, ytr)
    pet = et.predict_proba(Xval)[:, 1]
    pet_ = (pet > 0.5).astype(int)

    print("ET Accuracy: {} - Log Loss: {}".format(accuracy_score(yval, pet_), log_loss(yval, pet)))

    lr1 = make_pipeline(StandardScaler(), LogisticRegression())
    lr1.fit(Xtr, ytr)
    plr1 = lr1.predict_proba(Xval)[:, 1]
    plr1_ = (plr1 > 0.5).astype(int)

    print("LR StdScaler Accuracy: {} - Log Loss: {}".format(accuracy_score(yval, plr1_), log_loss(yval, plr1)))

    lr2 = make_pipeline(MinMaxScaler(), LogisticRegression())
    lr2.fit(Xtr, ytr)
    plr2 = lr2.predict_proba(Xval)[:, 1]
    plr2_ = (plr2 > 0.5).astype(int)

    print("LR MinMax Accuracy: {} - Log Loss: {}".format(accuracy_score(yval, plr2_), log_loss(yval, plr2)))

    second_level[ts, 0] = prf
    second_level[ts, 1] = pet
    second_level[ts, 2] = plr1
    second_level[ts, 3] = plr2

    print()

# fatores de diversidade

for tr, ts in kf.split(X, y):
    Xtr, Xval = second_level[tr], second_level[ts]
    ytr, yval = y.iloc[tr], y.iloc[ts]

    lr_stack = LogisticRegression(C=1.)
    lr_stack.fit(Xtr, ytr)
    plr_stack = lr_stack.predict_proba(Xval)[:, 1]
    plr_stack_ = (plr_stack > 0.5).astype(int)

    print("Stack Accuracy: {}  Log loss: {}".format(accuracy_score(yval, plr_stack_), log_loss(yval, plr_stack)))
    print()

pd.DataFrame(np.corrcoef(second_level.T))
