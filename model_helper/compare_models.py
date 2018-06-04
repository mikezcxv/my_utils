import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
data = pd.read_csv('./input/train.csv')
y_train = data['y']
X_train = data.drop('y', axis=1).select_dtypes(include=[np.number])
scores_100_trees = np.array([])
scores_110_trees = np.array([])
for i in range(10):
    fold = KFold(n_splits=5, shuffle=True, random_state=i)
    scores_100_trees_on_this_split = cross_val_score(
                                             estimator=xgb.XGBRegressor(
                                                        n_estimators=100),
                                             X=X_train, y=y_train,
                                             cv=fold, scoring='r2')
    scores_100_trees = np.append(scores_100_trees,
                                 scores_100_trees_on_this_split)
    scores_110_trees_on_this_split = cross_val_score(
                                            estimator=xgb.XGBRegressor(
                                                        n_estimators=110),
                                            X=X_train, y=y_train,
                                            cv=fold, scoring='r2')
    scores_110_trees = np.append(scores_110_trees,
                                 scores_110_trees_on_this_split)
ttest_rel(scores_100_trees, scores_110_trees)

# Мы получим следующее:

# Ttest_relResult(statistic=5.9592780076048273, pvalue=2.7039720009616195e-07)

# Статистика положительна, это значит, что числитель дроби положителен и соответственно среднее X1n больше среднего X2n. Вспомним,
# что мы максимизируем целевую метрику R2 и поймем, что первый алгоритм лучше, то есть 110 деревьев на этих данных проигрывают 100.
# P-value значительно меньше 0.05, поэтому мы смело отвергаем нулевую гипотезу, утверждая что модели существенно различны. Попробуйте посчитать mean() и std() для scores_100_trees и scores_110_trees и вы убедитесь, что просто сравнение средних здесь не катит (отличия будут порядка std).
# А между тем модели отличаются, причем отличаются очень сильно, и t-критерий Стьюдента для связанных выборок помог нам это показать.

# https://habrahabr.ru/company/ods/blog/336168/


# Example of tuning of single parameter
def optimize_param(X_train, y_train, n_folds=5):
    t_stats, n_trees = [], []
    for j in range(50, 100):
        current_score = np.array([])
        for i in range(10):
            fold = KFold(n_splits=n_folds, shuffle=True, random_state=i)
            scores_on_this_split = cross_val_score(
                  estimator=xgb.XGBRegressor(n_estimators=j),
                  X=X_train, y=y_train, cv=fold, scoring='r2')
            current_score = np.append(current_score, scores_on_this_split)
        t_stat, p_value = ttest_rel(current_score, scores_100_trees)
        t_stats.append(t_stat)
        n_trees.append(j)
    plt.plot(n_trees, t_stats)
    plt.xlabel('n_estimators')
    plt.ylabel('t-statistic')
