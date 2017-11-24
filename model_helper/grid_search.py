import pandas as pd
import sns


def analyze_params_impact(grid_scores, exclude_columns=['nthread', 'seed', 'silent'], debug=False):
    res = []
    for raw in grid_scores:
        info = raw._asdict()
        if debug:
            print(info['cv_validation_scores'], info['mean_validation_score'], info['parameters'])

        info['parameters']['mean_score'] = info['mean_validation_score']
        for c in exclude_columns:
            if c in info['parameters'].keys():
                del info['parameters'][c]

        res.append(info['parameters'])

    return pd.DataFrame(res)


def show_param_corr(grid_scores):
    df_params = analyze_params_impact(grid_scores)
    sns.heatmap(df_params.corr(), cmap='bwr')


# Draft/ bootstrap for Grid search
def get_optimal_params(df, target, oos=.8):
    from sklearn.model_selection import RandomizedSearchCV
    import math
    import xgboost as xgb

    tr = df.iloc[:math.ceil(len(df) * oos), :]
    df_train, target_train = tr.ix[:, tr.columns != target], tr.ix[:, target]

    xgb_model = xgb.XGBClassifier()

    parameters = {'nthread': [4],
                  'objective': ['binary:logistic'],
                  'scale_pos_weight': [1, 20, 41.7],
                  'learning_rate': [0.05],
                  'max_depth': [3, 4, 5],
                  'min_child_weight': [1, 7, 11, 25, 40],
                  'silent': [1],
                  'subsample': [0.5, 0.6, 0.7, 0.8, 0.85],
                  'colsample_bytree': [0.3, 0.4, 0.5, 0.7, 0.9],
                  'n_estimators': [300],
                  'reg_lambda': [0, 10, 50, 80, 200],
                  'gamma': [0, .5, 1, 3, 5, 10, 15],
                  'seed': [42]}

    clf = RandomizedSearchCV(xgb_model, parameters, cv=5, scoring='roc_auc', verbose=10, refit=True, n_iter=50)

    clf.fit(df_train, target_train)

    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    return clf


# TODO draft
# def improved_bayesian():
#     import xgboost as xgb
#     from xgboost.sklearn import XGBRegressor
#     from xgboost import plot_tree
#     import numpy as np
#     import matplotlib
#     import pandas as pd
#     matplotlib.use('agg')
#     import matplotlib.pyplot as plt
#     from sklearn.model_selection import cross_val_score
#     from bayes_opt import BayesianOptimization
#
#     def wrap_cv(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha, reg_lambda,
#                 max_delta_step=0,
#                 debug=False, num_folds=6, max_boost_rounds=300, verbose_eval=False, count_extra_run=3,
#                 train_test_penalize=0.05, count_extra_folds=1):
#         # Data structure in which to save out-of-folds preds
#         early_stopping_rounds = 30
#
#         params['min_child_weight'] = int(min_child_weight)
#         params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
#         params['max_depth'] = int(max_depth)
#         params['subsample'] = max(min(subsample, 1), 0)
#         params['gamma'] = max(gamma, 0)
#         params['alpha'] = max(alpha, 0)
#         params['reg_lambda'] = max(reg_lambda, 0)
#         params['max_delta_step'] = max(max_delta_step, 0)
#         # 'max_delta_step' : max_delta_step.astype(int),
#
#
#         #     , target_train = tr.ix[:, tr.columns != target], tr.ix[:, target]
#
#         results = {'test-auc-mean': [], 'test-auc-std': [], 'train-auc-mean': [],
#                    'train-auc-std': [], 'num_rounds': []}
#         iter_cv_result = []
#         for i in range(count_extra_run):
#             verb = verbose_eval if i < 2 else None
#             c_e_f = i % (count_extra_folds + 1) if count_extra_folds else 0
#
#             iter_cv_result.append(xgb.cv(dict(params, silent=1, seed=i + 1), dtrain,
#                                          num_boost_round=max_boost_rounds,
#                                          early_stopping_rounds=early_stopping_rounds,
#                                          verbose_eval=verb, show_stdv=False,
#                                          metrics={'auc'}, stratified=False,
#                                          nfold=num_folds + c_e_f))
#
#             results['num_rounds'].append(len(iter_cv_result[i]))
#             t = iter_cv_result[i].iloc[results['num_rounds'][i] - 1, :]
#
#             for c in ['test-auc-mean', 'test-auc-std', 'train-auc-mean', 'train-auc-std']:
#                 results[c].append(t[c])
#
#         num_boost_rounds = np.mean(results['num_rounds'])
#
#         # Show results
#         res = []
#         for c in ['train-auc-mean', 'test-auc-mean', 'train-auc-std', 'test-auc-std', 'num_rounds']:
#             factor = 100 if c != 'num_rounds' else 1
#
#             res.append({'type': c,
#                         'val': round(np.mean(results[c]) * factor, 2),
#                         'std': round(np.std(results[c]) * factor, 2),
#                         'min': round(np.min(results[c]) * factor, 2),
#                         'max': round(np.max(results[c]) * factor, 2),
#                         'median': round(np.median(results[c]) * factor, 2)
#                         })
#
#         if debug:
#             print('Best iteration:', num_boost_rounds)
#
#         r = pd.DataFrame(res)[['type', 'val', 'std', 'min', 'max', 'median']]
#
#         train_val = float(r.loc[r['type'] == 'train-auc-mean']['val'])
#         test_val = float(r.loc[r['type'] == 'test-auc-mean']['val'])
#         return test_val - (train_val - test_val) * train_test_penalize
#
#     df_raw = pd.read_csv('~/Documents/2017/model1/fraud/prod_clean/data/prepared_data_snapshot.csv')
#     # df_raw.fillna(-1, inplace=True)
#
#     target, pk = 'isfraud', 'applicationid'
#     oos, l = .8, len(df_raw)
#
#     data = df_raw.loc[:, df_raw.columns != target]
#     y = df_raw.loc[:, target]
#     # ?do we need transformation
#     # data = (data - data.mean()) / (data.max() - data.min())
#
#     train, labels = data.iloc[:round(l * oos), ], y.iloc[:round(l * oos), ]
#     test, target_test = data.iloc[round(l * oos):, ], y.iloc[round(l * oos):, ]
#
#     dtrain = xgb.DMatrix(train, labels)
#
#     num_rounds = 250
#     random_state = 42
#     num_iter = 30
#     # number of initialization points is at least 5, and 10 is even better
#     # https://www.kaggle.com/c/allstate-claims-severity/discussion/25905
#     init_points = 7
#     init_params = {'eta': 0.05, 'silent': 1, 'eval_metric': 'auc', 'verbose_eval': True,
#                    'seed': random_state, 'objective': 'binary:logistic', 'scale_pos_weight': 7.0}
#
#     params = init_params
#     xgbBO = BayesianOptimization(wrap_cv, {'min_child_weight': (20, 190),
#                                            'colsample_bytree': (0.15, 0.75),
#                                            'max_depth': (2, 4),
#                                            'subsample': (0.45, 0.9),
#                                            'gamma': (.25, 3),
#                                            'alpha': (0, 10),
#                                            'reg_lambda': (10, 100),
#                                            'max_delta_step': (0, 10)
#                                            })
#
#     xgbBO.maximize(init_points=init_points, n_iter=num_iter)
#
#     # 34 | 01m30s |   68.20050 |   10.0000 |             0.7500 |    3.0000 |      2.1095 |            63.6022 |      24.5430 |      0.4869 |
#     # 36 | 01m33s |   68.23650 |    6.8881 |             0.7500 |    2.9618 |      3.5842 |           136.5231 |      68.4368 |      0.9000 |

# def draft_bayes_opt():
#     from bayes_opt import BayesianOptimization
#
#     def xgb_evaluate(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha):
#         params['min_child_weight'] = int(min_child_weight)
#         params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
#         params['max_depth'] = int(max_depth)
#         params['subsample'] = max(min(subsample, 1), 0)
#         params['gamma'] = max(gamma, 0)
#         params['alpha'] = max(alpha, 0)
#
#         #     , target_train = tr.ix[:, tr.columns != target], tr.ix[:, target]
#
#         dtrain = xgb.DMatrix(df_train, target_train)
#         cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=5,
#                            seed=random_state, callbacks=[xgb.callback.early_stop(30)])
#
#         # TODO Redefine with own cv avg
#
#         return -cv_result['test-auc-mean'].values[-1]
#
#     num_rounds = 300
#     random_state = 42
#     num_iter = 10
#     init_points = 5
#     params = {'eta': 0.1, 'silent': 1, 'eval_metric': 'auc', 'verbose_eval': True, 'seed': random_state}
#
#     xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
#                                                 'colsample_bytree': (0.1, 1),
#                                                 'max_depth': (3, 4, 5),
#                                                 'subsample': (0.5, 1),
#                                                 'gamma': (0, 10),
#                                                 'alpha': (0, 10),
#                                                 })
#
#     xgbBO.maximize(init_points=init_points, n_iter=num_iter)

