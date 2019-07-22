import numpy as np
from sklearn.model_selection import KFold

# b
# im in next(train_dataset.__getitem__())
hist_len = 768

train_df = train_dataset.data.copy()
train_df = pd.concat([train_df, pd.DataFrame(), pd.DataFrame(np.zeros((len(train_df), hist_len)))], axis=1)
for i in tqdm(range(len(train_df))):
    h = train_dataset.__getitem__(i)['image'].histogram()
    for j, h in enumerate(h):
        train_df.at[i, j] = h

test_df = test_dataset.data.copy()
test_df = pd.concat([test_df, pd.DataFrame(), pd.DataFrame(np.zeros((len(test_df), hist_len)))], axis=1)
for i in tqdm(range(len(test_df))):
    h = test_dataset.__getitem__(i)['image'].histogram()
    for j, h in enumerate(h):
        test_df.at[i, j] = h


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS)
kf.get_n_splits(train_df)

clfs, metr = [], []

# train_df[train_df.columns[2:]], train_df[target]
for train_index, test_index in kf.split(train_df):
    X_train, X_test = train_df[train_df.columns[2:]].iloc[train_index], train_df[train_df.columns[2:]].iloc[test_index]
    y_train, y_test = train_df[target][train_index], train_df[target][test_index]
    #     print("TRAIN:", X_train.shape, "TEST:", X_test.shape)

    clf = LogisticRegression(C=2, random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=50).fit(
        X_train.values, y_train)
    clfs.append(clf)
    pr = clf.predict_proba(X_test.values)
    #     , class_weight = {0: 1, 2: 2, 1: 5, 4: 5, 3: 5}
    # , dual=False, C=1.0, solver=’warn’, max_iter=100, l1_ratio=None

    #     r = np.zeros(pr.shape)
    #     r += pr
    #     np.argmax(r / N_SPLITS, axis=1)

    print(stats.pearsonr(np.argmax(pr, axis=1), y_test))
    metr.append(stats.pearsonr(np.argmax(pr, axis=1), y_test)[0])

print(np.round(np.mean(metr), 5), np.round(np.std(metr), 5))

# train_df[['diagnosis', 4]].corr()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

target = 'diagnosis'

X_train, X_test, y_train, y_test = train_test_split(train_df[train_df.columns[2:]], train_df[target], test_size=0.05, random_state=42)
clf = LogisticRegression(C=1, random_state=42, solver='lbfgs', multi_class='multinomial').fit(X_train.values, y_train)
pr = clf.predict(X_test.values)
# , class_weight = {0: 1, 2: 2, 1: 3, 4: 3, 3: 4}
# , dual=False, C=1.0, fit_intercept=True, solver=’warn’, max_iter=100, l1_ratio=None

print(stats.pearsonr(pr, y_test))

# # clf.predict_proba(X[:2, :])
# clf.score(X, y)