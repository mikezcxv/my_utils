import xgboost as xgb
from xgboost.sklearn import XGBClassifier
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
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from model_helper.visualize import *
import random
import pickle
from tqdm import tqdm
import xgbfir
import operator


def split_into_folds(df, target, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    res = []
    for big_ind, small_ind in skf.split(np.zeros(len(df)), df[target]):
        res.append((df.iloc[big_ind], df.iloc[small_ind]))


class RunXGB:
    def __init__(self, xgb_params, df, df_train, target_train, df_test, target_test, id_test, target, pk,
                 num_of_test_splits=5):

        if 'scale_pos_weight' in xgb_params:
            bad_ratio = df.shape[0] / df[target].sum()
            if abs(bad_ratio - xgb_params['scale_pos_weight']) > 0.05:
                raise ValueError('!Always check scale_pos_weight on new dataset\n',
                                 'Param is %.10f but actual is %.10f' %
                                 (xgb_params['scale_pos_weight'], bad_ratio))

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

        results = {'test-auc-mean': [], 'test-auc-std': [], 'train-auc-mean': [], 'train-auc-std': [],
                   'num_rounds': []}
        iter_cv_result = []
        for i in tqdm(range(count_extra_run)):
            verb = verbose_eval if i < 2 else None
            iter_cv_result.append(xgb.cv(dict(self.xgb_params, silent=1, seed=i + 1), self.dtrain,
                                         num_boost_round=max_boost_rounds,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose_eval=verb, show_stdv=False,
                                         metrics={'auc'}, stratified=False, nfold=num_folds))

            results['num_rounds'].append(len(iter_cv_result[i]))
            t = iter_cv_result[i].ix[results['num_rounds'][i] - 1, :]

            for c in ['test-auc-mean', 'test-auc-std', 'train-auc-mean', 'train-auc-std']:
                results[c].append(t[c])

        num_boost_rounds = np.mean(results['num_rounds'])

        # Show results
        res = []
        for c in ['train-auc-mean', 'test-auc-mean', 'train-auc-std', 'test-auc-std', 'num_rounds']:
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

        return int(num_boost_rounds), pd.DataFrame(res)[['type', 'val', 'std', 'min', 'max', 'median']]

    def get_max_boost_by_last_periods(self, debug=False, count_last_periods=3):
        #         kf = KFold(n_splits=self.num_of_test_splits)
        #         kf.get_n_splits(self.df_train)

        kf = KFold(n_splits=self.num_of_test_splits)
        l = kf.get_n_splits(self.df_train)
        g = kf.split(self.df_train)
        f = []
        for i in range(0, self.num_of_test_splits - count_last_periods):
            next(g)

        for i in range(0, count_last_periods):
            f.append(next(g))

        # kf.split(self.df_train)
        # for train_index, test_index in kf.split(self.df_train):

        cv_result = xgb.cv(dict(self.xgb_params, silent=1), self.dtrain, num_boost_round=1500,
                           early_stopping_rounds=50, verbose_eval=50, show_stdv=False,
                           metrics={'auc'}, stratified=False, nfold=self.num_of_test_splits,
                           folds=f)

        # cv_result[['train-logloss-mean', 'test-logloss-mean']].plot()
        # plt.show()
        num_boost_rounds = len(cv_result)
        if debug:
            print(cv_result)
            print(num_boost_rounds)

        return num_boost_rounds

    def load_model(self, model_full_path='data/fraud_model.dat'):
        return pickle.load(open(model_full_path, "rb"))

    def go_and_save_model(self, num_boost_round, p_o_s, target, show_graph=True, threshold_useless=3,
                          files_prefix='', debug=True, show_auc=True,
                          model_full_path='data/fraud_model.dat', count_extra_run=1):
        params = dict(self.xgb_params, silent=1)

        df = self.df.copy()
        df = df.iloc[:math.ceil(len(df) * p_o_s), :]

        df_train = df.ix[:, df.columns != target]
        target_train = df.ix[:, target]
        df_train.drop([self.pk], axis=1, inplace=True)
        dtrain = xgb.DMatrix(df_train, target_train)

        if debug:
            print(df_train.shape)

        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        if count_extra_run:
            for i in tqdm(range(count_extra_run)):
                extra_model = xgb.train(dict(params, seed=i + 1), dtrain, num_boost_round=num_boost_round)
                pickle.dump(extra_model, open(model_full_path + '_seed' + str(i), "wb"))

        # save model to file
        pickle.dump(model, open(model_full_path, "wb"))

        files_prefix = '_' + files_prefix

        if debug:
            print("Using %d features" % len(model.feature_names))
            print(model.get_fscore(), len(model.get_fscore()))

        scores = model.get_fscore()
        list_useless = []
        almost_useless = []
        for fn in model.feature_names:
            if fn not in scores:
                list_useless.append(fn)
            elif scores[fn] < threshold_useless:
                almost_useless.append(fn)

        if debug:
            print("List useless:", list_useless)
            print("List almost useless:", almost_useless)

        if show_graph:
            fig, ax = plt.subplots(1, 1, figsize=(8, 16))
            xgb.plot_importance(model, height=0.5, ax=ax)
            plt.show()
            plt.savefig('imgs/all_data' + files_prefix + 'features.png')

    #  TODO compare!
    def go(self, num_boost_round, show_graph=True, threshold_useless=3, files_prefix='', debug=True, show_auc=True,
           count_extra_run=None, save_features_info=True, save_averaged=False,
           step_without_bagging=3, seed_shift=0):
        params = dict(self.xgb_params, silent=1, seed=0 + seed_shift)
        model = xgb.train(params, self.dtrain, num_boost_round=num_boost_round)
        files_prefix = '_' + files_prefix
        features_interaction_file = 'data/' + files_prefix + '_features_info.xlsx'

        # print(model.feature_names)
        features_scores = model.get_fscore()
        if debug:
            print("Using %d features" % len(model.feature_names))
            print(features_scores, len(features_scores))

        scores = model.get_fscore()
        list_useless, almost_useless = [], []
        for fn in model.feature_names:
            if fn not in scores:
                list_useless.append(fn)
            elif scores[fn] < threshold_useless:
                almost_useless.append(fn)

        if debug:
            print("List useless:", list_useless)
            print("List almost useless:", almost_useless)

        if show_graph:
            fig, ax = plt.subplots(1, 1, figsize=(8, 16))
            xgb.plot_importance(model, height=0.5, ax=ax)
            plt.show()
            plt.tight_layout()
            plt.savefig('imgs/' + files_prefix + 'features.png')

        y_pred = model.predict(self.dtest)

        if save_features_info:
            xgbfir.saveXgbFI(model, OutputXlsxFile=features_interaction_file,
                             MaxTrees=200, MaxHistograms=30)

        # _acc = accuracy_score(self.target_test, np.round(y_pred))
        _roc = roc_auc_score(self.target_test, y_pred)

        predictions_extra_run = []
        df_preds = pd.DataFrame({'pred_first': y_pred})

        if count_extra_run:
            rocs = [_roc]
            for i in tqdm(range(count_extra_run)):

                # self.df_train, self.target_train - split
                if (i + 1) % step_without_bagging == 0:
                    dtrain = self.dtrain
                else:
                    df_train_tmp = self.df_train.sample(frac=1, replace=True, random_state=i + 1 + seed_shift)
                    target_train_tmp = self.target_train.ix[df_train_tmp.index]
                    dtrain = xgb.DMatrix(df_train_tmp, target_train_tmp)

                extra_model = xgb.train(dict(params, seed=i + 1 + seed_shift), dtrain, num_boost_round=num_boost_round)
                extra_y_pred = extra_model.predict(self.dtest)
                rocs.append(roc_auc_score(self.target_test, extra_y_pred))
                predictions_extra_run.append(extra_y_pred)
                df_preds['pred_' + str(i) + '_is_full_%s' % str(int(i % step_without_bagging))] = extra_y_pred

            if save_averaged:
                if debug:
                    print('Results are averaged')
                for prediction in predictions_extra_run:
                    y_pred += prediction

                y_pred /= count_extra_run + 1

            if debug:
                print('Avg auc: %.2f +- %.2f; [%.2f .. %.2f]; range: %.2f; median: %.2f' %
                      (np.mean(rocs) * 100, np.std(rocs) * 100, np.min(rocs) * 100, np.max(rocs) * 100,
                       (np.max(rocs) - np.min(rocs)) * 100, np.median(rocs) * 100))
                print('Averaged model auc: %.2f' % (roc_auc_score(self.target_test, y_pred) * 100))

        if show_auc:
            draw_roc_curve(y_pred, self.target_test, files_prefix + '_auc.png', 'imgs', _roc)

        # print("Accuracy on hold-out: %.2f" % _acc)
        if count_extra_run is None and debug:
            print("Roc on test: %.2f (one run)" % (_roc * 100))

        if debug:
            print('Features info: ' + 'imgs/' + files_prefix + 'features.png')
            print('Distribution: imgs/' + files_prefix + 'distributions.png')
            print('Scores: data/' + files_prefix + 'result.csv')
            print('Features interactions: ' + features_interaction_file)

        result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})

        if show_graph:
            draw_distributions(result, self.target, files_prefix + 'distributions.png', 'imgs')

        result.to_csv('data/' + files_prefix + 'result.csv', index=False)
        return result, df_preds, {'used': features_scores, 'useless': list_useless,
                                  'almost_useless': almost_useless}

    # def go(self, num_boost_round, show_graph=True, threshold_useless=3, files_prefix='',
    #        debug=True, show_auc=True):
    #     params = dict(self.xgb_params, silent=1)
    #     model = xgb.train(params, self.dtrain, num_boost_round=num_boost_round)
    #
    #     if debug:
    #         print("Using %d features" % len(model.feature_names))
    #         # print(model.feature_names)
    #         print(model.get_fscore())
    #         print(len(model.get_fscore()))
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
    #         print("Useless:", list_useless)
    #         print("Almost useless:", almost_useless)
    #
    #     if show_graph:
    #         fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    #         xgb.plot_importance(model, height=0.5, ax=ax)
    #         plt.show()
    #         plt.savefig('imgs/' + files_prefix + '_features.png')
    #
    #     y_pred = model.predict(self.dtest)
    #     _acc = accuracy_score(self.target_test, np.round(y_pred))
    #     _roc = roc_auc_score(self.target_test, y_pred)
    #
    #     # false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
    #     # may differ
    #     # print auc(false_positive_rate, true_positive_rate)
    #
    #     if show_auc:
    #         draw_roc_curve(y_pred, self.target_test, files_prefix + '_auc.png', 'imgs', _roc)
    #
    #     print("Auc on hold-out: %.3f" % _roc, "Acc on hold-out: %.2f" % _acc)
    #
    #     result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})
    #
    #     if show_graph:
    #         draw_distributions(result, self.target, files_prefix + 'distributions.png', 'imgs')
    #
    #     result.to_csv('data/' + files_prefix + 'result.csv', index=False)
    #     return _roc

    def go_multi(self, num_boost_round, columns_sets, show_graph=True, threshold_useless=3,
                 files_prefix='', debug=True, show_auc=True, count_rounds=10, min_auc=0.660):

        y_preds = []
        files_prefix = '_' + files_prefix

        med_result = pd.DataFrame({'id': self.id_test, 'realVal': self.target_test})

        for j in range(0, count_rounds):
            # For random sampling
            # sub_cols = random.sample(set(self.df_train.columns.values), 70)
            #             sub_cols = columns_sets[j]

            df_train_with_reap = self.df_train.sample(frac=1, replace=True, random_state=j, axis=0)
            target_train_with_reap = self.target_train.ix[df_train_with_reap.index]

            sub_cols = [x for x in columns_sets if (x not in [self.target, self.pk])]
            dtrain = xgb.DMatrix(df_train_with_reap[sub_cols], target_train_with_reap)
            dtest = xgb.DMatrix(self.df_test[sub_cols])

            params = dict(self.xgb_params, silent=1,
                          scale_pos_weight=len(target_train_with_reap) / target_train_with_reap.sum())

            cv_result = xgb.cv(params, dtrain, num_boost_round=120, early_stopping_rounds=50,
                               verbose_eval=50, show_stdv=False, metrics={'auc'}, stratified=False, nfold=5)

            test_cv = cv_result['test-auc-mean'].iloc[-1]
            print("Test cv: %.3f" % test_cv)

            if test_cv > min_auc:
                model = xgb.train(params, dtrain, num_boost_round=round(num_boost_round * 1.2))

                # print(model.feature_names)
                if debug:
                    print("Using %d features" % len(model.feature_names))
                    print(model.get_fscore())
                    print(len(model.get_fscore()))

                scores = model.get_fscore()
                list_useless = []
                almost_useless = []
                for fn in model.feature_names:
                    if fn not in scores:
                        list_useless.append(fn)
                    elif scores[fn] < threshold_useless:
                        almost_useless.append(fn)

                if debug:
                    print("List useless:")
                    print(list_useless)
                    print("List almost useless:")
                    print(almost_useless)

                if show_graph:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 16))
                    xgb.plot_importance(model, height=0.5, ax=ax)
                    plt.show()
                    plt.savefig('imgs/' + files_prefix + 'features.png')

                y_pred = model.predict(dtest)
                y_preds.append(y_pred)
                med_result['target_' + str(j)] = y_pred

        for i in range(0, len(y_preds) - 1):
            print(i)
            y_pred += y_preds[i]

        y_pred /= len(y_preds)

        _roc = roc_auc_score(self.target_test, y_pred)

        corr_matrix = med_result.corr()
        print(corr_matrix)
        plt.figure()
        sns.heatmap(corr_matrix, cmap='bwr')
        plt.savefig("imgs/" + files_prefix + "_corr_matrix_results.png")
        plt.clf()

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)
        # may differ
        # print auc(false_positive_rate, true_positive_rate)

        if show_auc:
            draw_roc_curve(y_pred, self.target_test, files_prefix + '_auc.png', 'imgs', _roc)

        # print("Accuracy on hold-out: %.2f" % _acc)
        print("Roc on hold-out: %.3f" % _roc)

        result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})

        # if show_graph:
        draw_distributions(result, self.target, files_prefix + 'distributions.png', 'imgs')

        result.to_csv('data/' + files_prefix + 'result.csv', index=False)
        return _roc

    def cv_xgb(self, xgb_params, train, y_train, num_boost_rounds=474,
               split_useless_bottom=2, debug=True, monitor_last_periods=3,
               random_state=42, runs_for_fold=1, plot_features_score=False):
        # stratified_split = StratifiedShuffleSplit(n_splits=self.num_of_test_splits, random_state=0)
        X = train.values
        y = y_train.values
        kf = KFold(n_splits=self.num_of_test_splits, random_state=random_state)
        kf.get_n_splits(X)

        i = 0
        list_useless, list_alm, l_f = {}, {}, {}
        # for train_index, test_index in stratified_split.split(X, y):
        #     Cont.df_res = df_train[[pk]].copy()
        #     Cont.df_res['scores'] = np.NAN
        #     Cont.df_res['real_y'] = np.NAN

        res = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if debug:
                print("  Set %d:" % i, end=' ')
            i += 1

            dtrain = xgb.DMatrix(X_train, y_train, feature_names=train.columns.values)
            dtest = xgb.DMatrix(X_test, feature_names=train.columns.values)

            _roc = []
            for j in range(0, max(1, runs_for_fold)):
                model = xgb.train(dict(xgb_params, silent=1, seed=i * 1000 + j),
                                  dtrain, num_boost_round=num_boost_rounds, verbose_eval=0)
                train_predictions = model.predict(dtest)
                _roc.append(roc_auc_score(y_test, train_predictions))

                if plot_features_score and j == 0:
                    try:
                        fig, ax = plt.subplots(1, 1, figsize=(8, 16))
                        xgb.plot_importance(model, height=0.5, ax=ax)
                        print(model.get_fscore())
                        model.dump_model('reports/dump.raw_fold_%d.txt' % i)
                        xgbfir.saveXgbFI(model,
                                         OutputXlsxFile='reports/features_importance_fold%d.xlsx' % i,
                                         MaxTrees=200, MaxHistograms=30)
                    except ValueError:
                        print('Scores are not available for this booster')

                    plt.show()

                    # draw_distributions(result, self.target, 'cv_distributions.png', 'imgs', inline=True)

            # Cont.df_res.ix[test_index, 'scores'] = train_predictions
            # Cont.df_res.ix[test_index, 'real_y'] = y_test

            # _acc = accuracy_score(y_test, np.round(train_predictions))
            _roc = np.mean(_roc)
            res.append(_roc)

            # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, clf.predict(dtest))
            # may differ
            # print(auc(false_positive_rate, true_positive_rate))

            # print(_acc)
            if debug:
                print(round(_roc * 100, 2))
            list_useless[i], list_alm[i] = report_useless_features(model, split_useless_bottom)
            l_f[i] = np.unique(list_useless[i] + list_alm[i]).tolist()
            # list(set(list_useless[i] + list_alm[i]))

            # acc_overall += _acc
        # acc_overall /= self.num_of_test_splits
        # print(predictions, original)

        # print("Avg accuracy %.2f " % acc_overall)
        # list_useless[1], list_useless[2], list_useless[3],
        useless = list_useless[self.num_of_test_splits]
        almost_u = l_f[self.num_of_test_splits]

        for h in range(1, monitor_last_periods + 1):
            useless = find_lists_intersection(useless, list_useless[self.num_of_test_splits - h])
            almost_u = find_lists_intersection(useless, l_f[self.num_of_test_splits - h])

        if debug:
            print("Avg auc: %.2f std %.2f" % (round(np.mean(res) * 100, 2), (round(np.std(res) * 100, 2))),
                  end="\n\n")
            print("Useless in all periods:", useless, end="\n\n")
            print("Almost useless(< 2 splits) in all periods:", almost_u)

        return [useless, almost_u]

    # With wait_useless_eliminate=True run until all useless are empty
    def cv_chain(self, max_rounds, max_cv_rounds=2, min_split_prun=0, debug=True, random_state=42,
                 monitor_last_periods=3, runs_for_fold=1):
        useless, tmp = self.cv_xgb(self.xgb_params, self.df_train, self.target_train,
                                   max_rounds, min_split_prun + 1,
                                   random_state=random_state,
                                   monitor_last_periods=monitor_last_periods,
                                   runs_for_fold=runs_for_fold)

        i = 0
        usefull_features = []
        while len(useless) and i < max_cv_rounds - 1:
            i += 1
            try:
                df_train_copy
            except NameError:
                df_train_copy = self.df_train.copy()
                if min_split_prun > 0:
                    useless = tmp

            df_train_copy.drop(useless, axis=1, inplace=True)
            usefull_features = df_train_copy.columns.values
            if debug:
                print("Full CV round: %d; Features left: %d" % (i, len(df_train_copy.columns.values)))

            if min_split_prun > 0:
                tmp, useless = self.cv_xgb(self.xgb_params, df_train_copy, self.target_train,
                                           max_rounds, min_split_prun + 1,
                                           random_state=random_state,
                                           monitor_last_periods=monitor_last_periods,
                                           runs_for_fold=runs_for_fold)
            else:
                useless, almos_u = self.cv_xgb(self.xgb_params, df_train_copy, self.target_train,
                                               max_rounds,
                                               random_state=random_state,
                                               monitor_last_periods=monitor_last_periods,
                                               runs_for_fold=runs_for_fold)

        return useless, usefull_features

    # TODO finish
    def do_score(self, num_boost_round, show_graph=True, threshold_useless=3, files_prefix='',
                 debug=True, show_auc=True):
        y_preds = []
        files_prefix = '_' + files_prefix

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(target_test, y_pred)

        columns_sets = [core_columns, only_columns_v1, only_columns_adjusted, only_columns_last]

        for j in range(0, len(columns_sets)):
            sub_cols = columns_sets[j]
            sub_cols = [x for x in sub_cols if (x not in [target, pk])]
            dtrain = xgb.DMatrix(self.df_train[sub_cols], self.target_train)
            dtest = xgb.DMatrix(self.df_test[sub_cols])

            params = dict(self.xgb_params, silent=1)
            model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

            # print(model.feature_names)
            if debug:
                print("Using %d features" % len(model.feature_names))
                print(model.get_fscore())
                print(len(model.get_fscore()))

            scores = model.get_fscore()
            list_useless = []
            almost_useless = []
            for fn in model.feature_names:
                if fn not in scores:
                    list_useless.append(fn)
                elif scores[fn] < threshold_useless:
                    almost_useless.append(fn)

            if debug:
                print("List useless:")
                print(list_useless)
                print("List almost useless:")
                print(almost_useless)

            y_pred = model.predict(dtest)
            y_preds.append(y_pred)

        for i in range(0, len(y_preds) - 1):
            print(i)
            y_pred += y_preds[i]

        y_pred /= len(y_preds)

        result = pd.DataFrame({'id': self.id_test, self.target: y_pred, 'realVal': self.target_test})

        if show_graph:
            plt.hist(result.loc[self.target], 100, alpha=0.5, label='All')
            plt.legend(loc='upper right')
            plt.savefig('imgs/' + files_prefix + '_final_distributions.png')

        result.to_csv('imgs/' + files_prefix + 'result.csv', index=False)
        return _roc

    @staticmethod
    def wrap_cv(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha, reg_lambda,
                debug=False, num_folds=6, max_boost_rounds=300, verbose_eval=50, count_extra_run=3,
                train_test_penalize=0.05):
        # Data structure in which to save out-of-folds preds
        early_stopping_rounds = 30

        params['min_child_weight'] = int(min_child_weight)
        params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['max_depth'] = int(max_depth)
        params['subsample'] = max(min(subsample, 1), 0)
        params['gamma'] = max(gamma, 0)
        params['alpha'] = max(alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)

        #     , target_train = tr.ix[:, tr.columns != target], tr.ix[:, target]

        results = {'test-auc-mean': [], 'test-auc-std': [], 'train-auc-mean': [],
                   'train-auc-std': [], 'num_rounds': []}
        iter_cv_result = []
        for i in range(count_extra_run):
            verb = verbose_eval if i < 2 else None
            iter_cv_result.append(xgb.cv(dict(params, silent=1, seed=i + 1), dtrain,
                                         num_boost_round=max_boost_rounds,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose_eval=verb, show_stdv=False,
                                         metrics={'auc'}, stratified=False, nfold=num_folds))

            results['num_rounds'].append(len(iter_cv_result[i]))
            t = iter_cv_result[i].iloc[results['num_rounds'][i] - 1, :]

            for c in ['test-auc-mean', 'test-auc-std', 'train-auc-mean', 'train-auc-std']:
                results[c].append(t[c])

        num_boost_rounds = np.mean(results['num_rounds'])

        # Show results
        res = []
        for c in ['train-auc-mean', 'test-auc-mean', 'train-auc-std', 'test-auc-std', 'num_rounds']:
            factor = 100 if c != 'num_rounds' else 1

            res.append({'type': c,
                        'val': round(np.mean(results[c]) * factor, 2),
                        'std': round(np.std(results[c]) * factor, 2),
                        'min': round(np.min(results[c]) * factor, 2),
                        'max': round(np.max(results[c]) * factor, 2),
                        'median': round(np.median(results[c]) * factor, 2)
                        })

        if debug:
            print('Best iteration:', num_boost_rounds)

        r = pd.DataFrame(res)[['type', 'val', 'std', 'min', 'max', 'median']]

        train_val = float(r.loc[r['type'] == 'train-auc-mean']['val'])
        test_val = float(r.loc[r['type'] == 'test-auc-mean']['val'])
        return test_val - (test_val - train_val) * train_test_penalize

    @staticmethod
    def sort_features(f_list, print_score=True):
        sorted_features = sorted(f_list.items(), key=operator.itemgetter(1), reverse=True)
        for m in sorted_features:
            print("'" + m[0] + "',", end='')
            if print_score:
                print(str(m[1]) + ",")
            else:
                print()


class XgbCvScores:
    """
    Save cross validation scores for all folds
    """
    def __init__(self, xgb_params, df, target, pk, num_boost_rounds, num_of_test_splits=5, train_test_split=0.8):
        self.xgb_params = xgb_params
        self.target = target
        self.pk = pk
        self.df = df
        self.num_of_test_splits = num_of_test_splits
        self.num_boost_rounds = num_boost_rounds
        self.df.sort_values(pk, ascending=True, inplace=True)
        self.df = self.df.iloc[:math.ceil(len(self.df) * train_test_split), :]
        self.is_scored = False

    def split_into_folds(self, columns, random_state=42):
        skf = KFold(n_splits=self.num_of_test_splits, shuffle=False, random_state=random_state)
        res = []

        for big_ind, small_ind in skf.split(np.zeros(len(self.df)), self.df[self.target]):
            res.append((self.df[columns].iloc[big_ind], self.df[columns].iloc[small_ind]))

        return res

    def go(self, count_extra_run=3, save_folder='reports', save_file='oos_result'):
        self.df['scores'] = np.NaN

        acc_overall, roc_overall = 0, 0
        i = 0
        columns = [x for x in self.df.columns.values if x not in [self.pk]]
        columns2 = [x for x in columns if x not in [self.target]]
        splits = self.split_into_folds(columns)

        for train, test in splits:
            df_train, target_train = train.loc[:, columns2], train.loc[:, self.target]
            df_test, target_test = test.loc[:, columns2], test.loc[:, self.target]

            X_train, y_train = df_train.values, target_train.values
            X_test, y_test = df_test.values, target_test.values

            print("Set: %d" % i, ':')
            i += 1

            dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_train.columns.values)
            dtest = xgb.DMatrix(X_test, feature_names=df_train.columns.values)

            _rocs = []
            for j in tqdm(range(0, count_extra_run)):
                model = xgb.train(dict(self.xgb_params, silent=1, seed=42 + j), dtrain,
                                  num_boost_round=self.num_boost_rounds, verbose_eval=0)
                train_predictions = model.predict(dtest)

                if j == 0:
                    self.df.loc[test.index, 'scores'] = train_predictions

                _rocs.append(roc_auc_score(y_test, train_predictions))

            print(np.round(np.mean(_rocs) * 100, 2),
                  np.round(np.std(_rocs) * 100, 2),
                  np.round(_rocs, 4) * 100)
            roc_overall += np.mean(_rocs)

        roc_overall /= self.num_of_test_splits

        print("Avg roc %.2f " % (roc_overall * 100))

        result = pd.DataFrame({'id': self.df[self.pk], self.target: self.df['scores'], 'realVal': self.df[self.target]})
        result.to_csv(save_folder + '/' + save_file + '.csv', index=False)

        return result

    def show_stats(self, inline=True, imgs_folder='imgs', auc_img_file='oos_cv_auc',
                   distr_img_file='oos_distributions'):
        _roc = roc_auc_score(self.df[self.target], self.df['scores'])
        draw_roc_curve(self.df['scores'], self.df[self.target], auc_img_file + '.png',
                       imgs_folder, _roc, inline=inline)
        result = pd.DataFrame({'id': self.df[self.pk], self.target: self.df['scores'], 'realVal': self.df[self.target]})
        draw_distributions(result, self.target, distr_img_file + '.png', imgs_folder, inline=inline)



'''
    def final_scores(self, num_boost_round, show_graph=1, threshold_useless=3, files_prefix=''):
        df_train = self.df.ix[:, df.columns != self.target]
        target_train = df.ix[:, self.target]
        target_test = target_train
        id_test = df_train[self.pk]
        df_train.drop([self.pk], axis=1, inplace=True)
        df_test = df_train       
        files_prefix = '_' + files_prefix

        dtrain = xgb.DMatrix(df_train, target_train)
        dtest = xgb.DMatrix(df_test)

        params = dict(xgb_params, silent=1)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

        scores = model.get_fscore()
        list_useless = []
        almost_useless = []
        for fn in model.feature_names:
            if fn not in scores:
                list_useless.append(fn)
            elif scores[fn] < threshold_useless:
                almost_useless.append(fn)

        print("List useless:")
        print(list_useless)
        print("List almost useless:")
        print(almost_useless)

        y_pred = model.predict(dtest)
        _acc = accuracy_score(target_test, np.round(y_pred))
        _roc = roc_auc_score(target_test, y_pred)

        show_auc = True
        if show_auc:
            fpr, tpr, _ = roc_curve(target_test, y_pred)

            # Plot of a ROC curve for a specific class
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % _roc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig('imgs/' + provider + '_aucc_full.png')
            plt.clf()

        print("Accuracy on hold-out: %.2f" % _acc)
        print("Roc on hold-out: %.3f" % _roc)
        result = pd.DataFrame({'id': id_test, target: y_pred, 'realVal': target_test})

        # TODO
        show_graph = True
        if show_graph:
            plt.hist(result.loc[result.realVal == 1, target], 100, alpha=0.8, label='Bad', color="red")
            plt.hist(result.loc[result.realVal == 0, target], 100, alpha=0.5, label='Good')
            plt.legend(loc='upper right')
            # pyplot.show()
            plt.savefig('imgs/' + provider + '_distributions_full.png')

        # TODO change dir
        result.to_csv('imgs/' + provider + '_full_result.csv', index=False)
'''