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

