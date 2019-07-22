# train_df[['diagnosis', 4]].corr()
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

target = 'diagnosis'

X_train, X_test, y_train, y_test = train_test_split(train_df[train_df.columns[2:]], train_df[target], test_size=0.15,
                                                    random_state=42)

# , weight=y_train * 2 + 1
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Train the model
# parameters = {
#     'application': 'binary',
#     'objective': 'binary',
#     'metric': 'auc',
#     'is_unbalance': 'true',
#     'boosting': 'gbdt',
#     'num_leaves': 31,
#     'feature_fraction': 0.5,
#     'bagging_fraction': 0.5,
#     'bagging_freq': 20,
#     'learning_rate': 0.05,
#     'verbose': 0
# }

params = {
    "objective": "multiclass",
    "num_class": 5,
    "num_leaves": 60,
    "max_depth": -1,
    "learning_rate": 0.01,
    "bagging_fraction": 0.8,  # subsample
    "feature_fraction": 0.7,  # colsample_bytree
    "bagging_freq": 5,  # subsample_freq
    "bagging_seed": 42
    #         "sample_weight": [1, 2, 1, 2, 3]
    #     {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}
}

model = lgb.train(params, train_data, valid_sets=test_data, num_boost_round=500, early_stopping_rounds=20,
                  verbose_eval=5)

pr = model.predict(X_test.values)
# , class_weight = {0: 1, 2: 2, 1: 3, 4: 3, 3: 4}
# , dual=False, C=1.0, fit_intercept=True, solver=’warn’, max_iter=100, l1_ratio=None

# print(stats.pearsonr(pr, y_test))
# print(stats.pearsonr(np.argmax(pr, axis=1), y_test))

for i in range(0, 5):
    print(roc_auc_score(np.where(y_test == i, 1, 0), np.where(pr[:, i] > .5, 1, 0)))
print(stats.pearsonr(np.argmax(pr, axis=1), y_test))

# [340]	valid_0's multi_logloss: 0.731805

# # clf.predict_proba(X[:2, :])
# clf.score(X, y)