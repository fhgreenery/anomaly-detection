import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats

df = np.loadtxt('data/194444.txt',dtype=int,delimiter='\t')
print(df)
rng = np.random.RandomState(42)
print(rng)
n_samples=17  #样本总数

# fit the model
clf = IsolationForest(max_samples=n_samples, random_state=rng, contamination=0.18,n_jobs=-1,behaviour="new")  #contamination为异常样本比例
clf.fit(df)
scores_pred = clf.decision_function(df)
print(scores_pred)
print(len(scores_pred))
threshold = stats.scoreatpercentile(scores_pred, 100 * 0.18)
y = clf.predict(df)
print(y)
