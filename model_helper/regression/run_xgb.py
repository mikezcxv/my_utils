import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import numpy as np
import pandas as pd
import math
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ggplot import *
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from model_helper.visualize import *
import random
import pickle
from tqdm import tqdm
import model_helper.common


class RunXGB:
    def __init__(self, xgb_params, df, df_train, target_train, df_test, target_test, id_test, target, pk,
                 num_of_test_splits=5):
        self.xgb_params = xgb_params
        self.target = target
        self.pk = pk
        self.df = df
        self.num_of_test_splits = num_of_test_splits
        self.df_train = df_train
        self.df_test = df_test
        self.target_train = target_train
        self.target_test = target_test
        self.dtrain = xgb.DMatrix(df_train, target_train)
        self.dtest = xgb.DMatrix(df_test)
        self.id_test = id_test

    def get_max_boost(self, debug=False, num_folds=6, max_boost_rounds=1500, verbose_eval=50, count_extra_run=1):
        # Data structure in which to save out-of-folds preds
        early_stopping_rounds = 30

        results = {'test-rmse-mean': [], 'test-rmse-std': [], 'train-rmse-mean': [], 'train-rmse-std': [],
                   'num_rounds': []}

        iter_cv_result = []
        for i in tqdm(range(count_extra_run)):
            verb = verbose_eval if i < 2 else None
            iter_cv_result.append(xgb.cv(dict(self.xgb_params, silent=1, seed=i + 1), self.dtrain,
                                         num_boost_round=max_boost_rounds,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose_eval=verb, show_stdv=False,
                                         metrics={'rmse'}, stratified=False, nfold=num_folds))

            results['num_rounds'].append(float(len(iter_cv_result[i])))
            t = iter_cv_result[i].ix[results['num_rounds'][i] - 1, :]

            for c in ['test-rmse-mean', 'test-rmse-std', 'train-rmse-mean', 'train-rmse-std']:
                results[c].append(t[c])

        num_boost_rounds = np.mean(results['num_rounds'])

        # Show results
        res = []
        for c in ['train-rmse-mean', 'test-rmse-mean', 'train-rmse-std', 'test-rmse-std', 'num_rounds']:
            factor = 100 if c != 'num_rounds' else 1

            res.append({'type': c,
                        'val': round(np.mean(results[c]) * factor, 2),
                        'std': round(np.std(results[c]) * factor, 2),
                        'min': round(np.min(results[c]) * factor, 2),
                        'max': round(np.max(results[c]) * factor, 2),
                        'median': round(np.median(results[c]) * factor, 2)
                        })

            # print('Avg %s: %.2f +- %.2f; [%.2f .. %.2f]; range: %.2f; median: %.2f' %
            #       (c, np.mean(results[c]) * factor, np.std(results[c]) * factor,
            #        np.min(results[c]) * factor,
            #        np.max(results[c]) * factor, (np.max(results[c]) - np.min(results[c])) * factor,
            #        np.median(results[c]) * factor))

        if debug:
            # print('Results of the first round', iter_cv_result[0])
            print('Best iteration:', num_boost_rounds)

        return int(num_boost_rounds), pd.DataFrame(res)[['type', 'val', 'std', 'min', 'max', 'median']], res

    # def go(self, num_boost_round, show_graph=True, threshold_useless=3, files_prefix='', debug=True, show_auc=True,
    #        count_extra_run=None):
    #     params = dict(self.xgb_params, silent=1)
    #     model = xgb.train(params, self.dtrain, num_boost_round=num_boost_round)
    #     files_prefix = '_' + files_prefix
    #
    #     # print(model.feature_names)
    #     if debug:
    #         print("Using %d features" % len(model.feature_names))
    #         print(sort_dictionary(model.get_fscore()), len(model.get_fscore()))
    #
    #     scores = model.get_fscore()
    #     list_useless, almost_useless = [], []
    #     for fn in model.feature_names:
    #         if fn not in scores:
    #             list_useless.append(fn)
    #         elif scores[fn] < threshold_useless:
    #             almost_useless.append(fn)
    #
    #     if debug:
    #         print("List useless:", list_useless)
    #         print("List almost useless:", almost_useless)
    #
    #     if show_graph:
    #         fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    #         xgb.plot_importance(model, height=0.5, ax=ax)
    #         plt.show()
    #         plt.savefig('imgs/' + files_prefix + 'features.png')
    #
    #     y_pred = model.predict(self.dtest)
    #
    #     # ' R2:', round(r2_score(target_test, y_predicted), 3), '\n',
    #     # 'RMSLE: %.2f' % rmsle_vector(target_test, y_predicted), '\n',
    #     # 'MSE:', round(mean_squared_error(target_test, y_predicted), 2), '\n',
    #     # 'MAE:', round(median_absolute_error(target_test, y_predicted), 2), '\n'
    #
    #     _acc = accuracy_score(self.target_test, np.round(y_pred))
    #     _roc = roc_auc_score(self.target_test, y_pred)
    #
    #     if count_extra_run:
    #         rocs = []
    #         for i in tqdm(range(count_extra_run)):
    #             extra_model = xgb.train(dict(params, seed=i + 1), self.dtrain, num_boost_round=num_boost_round)
    #             extra_y_pred = extra_model.predict(self.dtest)
    #             rocs.append(roc_auc_score(self.target_test, extra_y_pred))
    #
    #         print('Avg auc: %.2f +- %.2f; [%.2f .. %.2f]; range: %.2f; median: %.2f' %
    #               (np.mean(rocs) * 100, np.std(rocs) * 100, np.min(rocs) * 100, np.max(rocs) * 100,
    #                (np.max(rocs) - np.min(rocs)) * 100, np.median(rocs) * 100))
    #
    #     if show_auc:
    #         draw_roc_curve(y_pred, self.target_test, files_prefix + '_auc.png', 'imgs', _roc)
    #
    #     # print("Accuracy on hold-out: %.2f" % _acc)
    #     print("Roc on hold-out: %.2f (one run)" % _roc * 10)
    #
    #     result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})
    #
    #     if show_graph:
    #         draw_distributions(result, self.target, files_prefix + 'distributions.png', 'imgs')
    #
    #     result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})
    #     result.to_csv('data/' + files_prefix + 'result.csv', index=False)
    #     return _roc