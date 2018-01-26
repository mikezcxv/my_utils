import math
import re
import gc
import scipy.stats as stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model_helper.Transform import *
from model_helper.stat_categorical import  *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# describe, show_na, analyze_na - overall
# compare_binary - for binary classification


def describe(df, pk, target):
    print(df.shape)
    print(df.describe())
    print(df.groupby(pk)[[target]].agg(['count', 'min', 'mean', 'max']))
    print(df.isnull().sum())


def show_na(df, percent=.1, debug=False):
    l = len(df)
    i = 0
    columns = []
    for c in df.columns.values:
        count_na = df[c].isnull().sum()
        if count_na / l >= percent:
            i += 1
            columns.append(c)
            if debug:
                print('%d[%.1f%%]  %s' % (count_na, (count_na / l) * 100, c))

    if debug:
        print('%d has >= than %.3f nan' % (i, percent))
    return i, percent, columns


def analyze_na(df, debug=True):
    show_na(df, debug=debug)
    l = [.2, .5, .7, .9, .95, .99, 1]
    count_columns = len(df.columns.values)
    for pt in l:
        i, percent, columns = show_na(df, percent=pt)
        if debug:
            print('%d [%.1f%%] has >= than %.3f nan' % (i, (i / count_columns) * 100, percent))

    return columns


def show_null(df):
    print(df.isnull().sum())


def group_bad_level(df, target, pk_name):
    return df[[pk_name, target]].groupby([target], as_index=False).agg(['count'])


def check_normality(data_list, dist='norm', debug=False):
    shapiro = None
    if dist == 'norm':
        shapiro = stats.shapiro(data_list)
    ks = stats.kstest(data_list, dist)
    anderson = stats.anderson(data_list, dist=dist)

    if debug:
        print("""For Anderson's: Critical values provided are for the following significance levels:
                 normal / exponenential      15%, 10%, 5%, 2.5%, 1%
                 logistic                    25%, 10%, 5%, 2.5%, 1%, 0.5%
                 Gumbel                      25%, 10%, 5%, 2.5%, 1%
        """)

    return {'shapiro': shapiro, 'ks': ks, 'anderson': anderson}


# TODO check
def show_binary(df, name):
    print("%s len: %d; count 1: %d; count 0: %d " %
          (name, len(df), len(df.loc[np.where(df == 1.0)]),
           len(df.loc[np.where(df == 0.0)])))


def find_low_variance(df, max_v=0.0001, is_print=True):
    res = set()
    for k, v in df.var().items():
        if v <= max_v:
            if is_print:
                print(k, v)
            res.add(k)

    return res


def find_equals(df, print_only_different=True):
    values = df.columns.values
    for i in range(0, len(values)):
        for j in range(0, len(values)):
            if j > i:
                is_equal = df[values[i]].equals(df[values[j]])
                if print_only_different:
                    if is_equal:
                        print("Pair: %s x %s" % (values[i], values[j]))
                else:
                    print("Pair: %s x %s: %d" % (values[i], values[j], is_equal))


def get_top_cor(df, threshold=0.95):
    cor = df.corr()
    indices = np.where(cor > threshold)
    return [(cor.index[x], cor.columns[y]) for x, y in zip(*indices) if x != y and x < y]


def find_correlated(df, column_name, threshold=0.5):
    """
    Find all columns that are highly correlated to the given
    :param df:
    :param column_name:
    :param threshold:
    :return:

    """
    for c in df:
        if df[c].dtype == 'object':
            continue

        sub_df = df.loc[~pd.isnull(df[c]) & ~pd.isnull(df[column_name])][[c, column_name]]

        # ?
        # if len(sub_df) < 100:
        #     continue

        try:
            corr = stats.pearsonr(sub_df[c], sub_df[column_name])
        except ValueError:
            print('Error with ', c)

        if abs(corr[0]) > threshold:
            print(c, corr, 'NA:', len(df) - len(sub_df))


def find_unstable(l, allowed_std=.025):
    s = np.std(l)
    left = np.min(l)
    right = np.max(l)

    return s > allowed_std or (right - .5) * (left - .5) < 0


def is_around_0(l, allowed_range=.02):
    left = np.min(l)
    right = np.max(l)
    return (right - left < allowed_range) and (math.fabs(right - .5) < allowed_range / 2)


def compare_binary(df, feature1, low_val=0, hi_val=1, target='y', ignore_identical=False):
    l = df.loc[df[target] == low_val]
    h = df.loc[df[target] == hi_val]

    mean_neg, mean_pos = round(np.mean(l[feature1]), 5), round(np.mean(h[feature1]), 5)

    to_print = 'Count low: ' + str(len(l)) + ' hi:' + str(len(h))
    to_print += '\nMean low: ' + str(mean_neg) + ' hi:' + str(mean_pos)
    count_pos, count_neg = len(h), len(l)

    #     print(l[feature1].isnull().sum(), len(l), h[feature1].isnull().sum(), len(h))
    if (l[feature1].isnull().sum()) == count_neg or (h[feature1].isnull().sum() == count_pos):
        return None, None, None, None, None, None

    r = stats.ks_2samp(l[feature1], h[feature1])
    is_signif = False
    if r.pvalue < 0.001:
        to_print += '\n\033[1;36m!!different distributions\x1b[0m'
        is_signif = True
    elif r.pvalue < 0.01:
        to_print += '\n\033[1;36mdifferent distributions\x1b[0m'
        is_signif = True
    elif r.pvalue < 0.1:
        to_print += '\nslightly different'
        is_signif = True
    else:
        to_print += '\n\x1b[31midentical\x1b[0m'

    if not ignore_identical or is_signif:
        #         print(to_print, r)
        1 == 1
    else:
        to_print = 'Ignore'
    # print('Ignore')

    return r, to_print, count_pos, count_neg, mean_neg, mean_pos


def get_ks_stat(df, target, neg_val=0, pos_val=1, max_runs=10000):
    i = 0
    agg = []

    for c in tqdm(df.columns.values):
        try:
            r, resolution, count_pos, count_neg, \
                mean_pos, mean_neg = compare_binary(df, c,
                                                    low_val=neg_val, hi_val=pos_val, target=target)

            if r:
                item = {'column': c, 'stat': round(r[0], 4), 'p': round(r[1], 7),
                           'count_pos': count_pos, 'count_neg': count_neg,
                            'mean_pos': mean_pos, 'mean_neg': mean_neg}
                agg.append(item)
            else:
                agg.append({'column': c, 'stat': None, 'p': None, 'count_pos': count_pos, 'count_neg': count_neg,
                           'mean_pos': mean_pos, 'mean_neg': mean_neg})

        except ValueError and TypeError:
            print('Can"t handle %s' % c)
        if i > max_runs:
            break
        i += 1

    return pd.DataFrame(agg)


def find_features_combs(df, gain_percent=1.4, na_threshold=0.98, seed=42, target_name='isfraud',
                        skip_columns=()):
    """
    find_combs(df_prepared2, gain_percent=2, na_threshold=0.95, seed=42)
    """
    gc.collect()
    fin = {}

    total = len(df) * .7
    res = {}

    # TODO make more rands with different seeds
    subs = df.sample(frac=.7, random_state=seed, replace=True)

    check_columns = df.columns.values
    check_columns = [x for x in check_columns if x not in skip_columns]

    for c in check_columns:
        # TODO check
        a = roc_auc_score(subs.loc[~pd.isnull(subs[c])][target_name],
                          subs.loc[~pd.isnull(subs[c])][c])

        # if not subs[c].isnull().values.any():
        #     a = roc_auc_score(subs[target_name], subs[c])
        #     # print("%0.3f: %s" % (a, c))
        res[c] = a

    functions = {
        '*': lambda x, y: x * y,
        '+**2': lambda x, y: x + y ** 2,
        '2**+': lambda x, y: x ** 2 + y,
        '+log': lambda x, y: x + np.log(y + 0.000001),
        '*log': lambda x, y: x * np.log(y + 0.000001),
        'log+': lambda x, y: np.log(x + 0.000001) + y,
        'log*': lambda x, y: np.log(x + 0.000001) * y,
        '/': lambda x, y: x / y,
        '/r': lambda x, y: y / x,
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        # '-r': lambda x, y: y - x,
    }

    i = 0
    for x1 in check_columns:
        j = 0
        for x2 in check_columns:
            if i > j and x1 in res.keys() and x2 in res.keys():
                for op, fn in functions.items():
                    r = fn(subs[x1], subs[x2])
                    r = r.replace([np.inf], 10000)
                    r = r.replace([-np.inf], -10000)
                    count_na = len(r.loc[pd.isnull(r)])
                    count_s = r.sum()

                    if ((total - count_na) / total < na_threshold) \
                            or count_s == 0:
                        continue

                    new_auc = roc_auc_score(subs.ix[~np.isnan(r)][target_name],
                                            r.ix[~np.isnan(r)])

                    if abs(new_auc - .5) > abs(res[x1] - .5) + gain_percent / 100 \
                            and abs(new_auc - .5) > abs(res[x2] - .5) + gain_percent / 100:
                        print("%.4f * %.4f -> %.4f %s %s %s %d" %
                              (res[x1], res[x2], new_auc, x1, op, x2, count_na))

                        k = "%s %s %s %d" % (x1, op, x2, count_na)
                        v = "%.4f * %.4f -> %.4f" % (res[x1], res[x2], new_auc)
                        if k in fin:
                            fin[k].append(v)
                        else:
                            fin[k] = [v]

                            # res[c] = a
            j += 1
        i += 1
    return fin


def show_binary_comparison_for_regression(df, feat_name, by_fea='y', left=0, right=1):
    print(df.groupby(feat_name, as_index=False)[by_fea]
          .agg(['count', 'mean', 'min', 'max']))

    set1 = df[df[feat_name] == right][by_fea]
    set2 = df[df[feat_name] == left][by_fea]
    print(stats.ks_2samp(set1, set2))

    original1, original2 = len(set1), len(set2)

    set1 = set1[~np.isnan(set1)]
    set2 = set2[~np.isnan(set2)]

    after_rm1, after_rm2 = len(set1), len(set2)

    if after_rm1 < original1:
        print('%d values removed from set 1!' % (original1 - after_rm1))

    if after_rm2 < original2:
        print('%d values removed from set 2!' % (original2 - after_rm2))

    print(stats.ttest_ind(set1, set2))


def iv2(df, column_name, y, min_in_group=30, debug=False, max_classes=200, return_default=-0.000001):
    info = df[column_name].value_counts().to_dict()
    s = 0

    if debug:
        print(info)

    c_bad_group = len(df.loc[~pd.isnull(df[column_name]) & (y == 1)]) + 0.0000000000001
    c_good_group = len(df.loc[~pd.isnull(df[column_name]) & (y == 0)]) + 0.0000000000001

    if debug:
        print("Bad in group: %d; Good in group: %d" % (c_bad_group, c_good_group))

    if (len(info) > max_classes) or (len(info) < 2):
        return return_default

    for k, v in info.items():
        bad_count = len(df.loc[(df[column_name] == k) & (y == 1)])
        good_count = len(df.loc[(df[column_name] == k) & (y == 0)])

        bad_rate = 0.0000000000001 + (bad_count / c_bad_group)
        good_rate = good_count / c_good_group
        woe = np.log(good_rate / bad_rate)
        if debug:
            print("Val: %s; In group: %d; Bad r: %.4f; Bad count: %d; Good r: %.4f; Goog c: %d; Woe: %.4f"
                  % (k, v, bad_rate, bad_count, good_rate, good_count, woe))
        if v > min_in_group:
            s += (good_rate - bad_rate) * woe

    if debug:
        print(s)
    return s


def features_info(df, split_ranges=(0, 1), target_name='y', skip_columns=(''), only_columns=(), result_prefix='',
                  set_na=-1):
    out_csv = True
    na_value = set_na
    df = Transform().cast_multi_train(df, na_val=set_na)

    # Handle categorical
    X = df.loc[:, df.columns != target_name]
    y = df.loc[:, target_name]

    X_chunks, y_chunks = {}, {}
    l = len(X)
    count_ranges = len(split_ranges) - 1
    for i in range(0, count_ranges):
        # print("From %.3f to %.3f" % (int(split_ranges[i] * l), int(split_ranges[i + 1] * l)))
        X_chunks[i] = X.iloc[int(split_ranges[i] * l): int(split_ranges[i + 1] * l), :]
        y_chunks[i] = y.iloc[int(split_ranges[i] * l): int(split_ranges[i + 1] * l)]

    count_skip = 0
    count_na = {}

    for i in range(count_ranges):
        _df = X_chunks[i]
        for c in _df.columns.values:
            count_na[str(i) + '_' + c] = len(_df.loc[np.isnan(_df[c])])
        _df.fillna(na_value, inplace=True)

    res = []
    df_csv = pd.DataFrame(columns=['Feature', 'Time period', 'Auc', 'IV', 'Count Groups', 'Count NA', 'Percent NA'])
    for c in X.columns.values:
        if c in skip_columns:
            continue

        if only_columns and c not in only_columns:
            continue
        f_roc = {}
        f_roc_clean = {}
        count_groups = {}
        _iv = {}
        _iv_clean = {}
        for i in range(count_ranges):
            _df = X_chunks[i]
            _y = y_chunks[i]
            count_groups[i] = len(_df[c].value_counts().to_dict())
            if count_groups[i] > 1:
                f_roc[i] = roc_auc_score(_y, _df[c])
                count_groups_na = len(_df.loc[~_df[c].isin([na_value])][c].value_counts().to_dict())

                if count_groups_na > 1:
                    if len(_y.loc[~_df[c].isin([na_value])].unique()) == 1:
                        f_roc_clean[i] = 1
                    else:
                        f_roc_clean[i] = roc_auc_score(_y.loc[~_df[c].isin([na_value])],
                                                       _df.loc[~_df[c].isin([na_value])][c])
                else:
                    f_roc_clean[i] = None
            else:
                f_roc[i] = -0.00001
                f_roc_clean[i] = -0.00001

            # TODO don't calculate IV for continuous vars
            _iv[i] = iv2(_df, c, _y)
            _iv_clean[i] = iv2(_df.loc[~_df[c].isin([na_value])], c, _y)
            # _iv = 0 # s = 1 / (1 + np.exp(-df_train[c])) # f_roc2 = roc_auc_score(target_train, s)

            # _ks = compare_binary(_df, 'SecondPaymentDelinquencyClass', target=c)

        flag = '\033[1;36m'
        if find_unstable(list(f_roc_clean.values())) or is_around_0(list(f_roc_clean.values())):
            flag = '\033[1;31m'

        # TODO
        '''   
        if ( math.fabs(f_roc - f_roc_test) > .02 ) or ( math.fabs(f_roc_test - .5) < .015 ) or ( math.fabs(f_roc - .5) < .01 ) :
              flag = '\033[1;31m'
        else:
              flag = '\033[1;36m'

        if ( math.fabs(f_roc_test - .5) > .05 ) or ( math.fabs(f_roc - .5) > .05 ):
             flag = flag + '!!'

        if ( math.fabs(f_roc_test - .5) > .8 ) or ( math.fabs(f_roc - .5) > .8 ):
             flag = flag + '!!!'
        '''

        report = ''
        for i in range(count_ranges):
            percent_na = round(count_na[str(i) + '_' + c] / len(X_chunks[i]), 2) * 100
            report += "PERIOD {}: Auc {:.5f}[{:.5f}]; IV :{:.5f}[{:.5f}]; Gr. {}; NA: {} {}%\n".format(
                i, f_roc[i], f_roc_clean[i], _iv[i], _iv_clean[i],
                count_groups[i], count_na[str(i) + '_' + c],
                percent_na)
            df_csv.loc[len(df_csv)] = [c, i, round(f_roc_clean[i], 6), round(_iv_clean[i], 6), count_groups[i],
                                       count_na[str(i) + '_' + c], percent_na]

            # res.append({
            #     "PERIOD {}: Auc {:.5f}[{:.5f}]; IV :{:.5f}[{:.5f}];".format(
            #         i,
            #         f_roc[i],
            #         f_roc_clean[i],
            #         _iv[i],
            #         _iv_clean[i],
            #         'groups': count_groups[i],
            #         'count_na': count_na[str(i) + '_' + c],
            #         'percent_na': percent_na
            # })

        print(c)
        print(flag + report + "\033[0m")
    # print(flag + "Auc %.5f[%.5f] -> %.5f[%.5f]; IV %.5f -> %.5f; Gr. %d -> %d; NA: %d\t%s \033[0m"
    #               % (f_roc, f_roc_clean, f_roc_test, f_roc_test_clean, _iv, _iv_test, count_groups, count_groups_test, count_na_train[c], c))

    if out_csv:
        df_csv.to_csv('imgs/' + result_prefix + 'top_features_8.csv', index=False)

    print("Skipped %d" % count_skip)


def find_high_variance(df, min_v=.002, is_print=True, return_scores=False):
    res = list()
    res2 = {}
    for k, v in df.var().items():
        if v >= min_v:
            if is_print:
                print(k, v)
            res.append(k)
            if return_scores:
                res2.update({k: v})

    return res2 if return_scores else res


# TODO
def show_statuses_info(df):
    c = df[['status', 'application_id']].groupby('status').agg(['count']) \
        ['application_id'].sort_values(['count'], ascending=False)
    c['p'] = c['count'] / len(df)
    c['p'] *= 100
    print(c)


def explore_var(df, name):
    l = len(df)
    count_groups = len(df[name].unique())
    count_na = df[name].isnull().sum()

    print('Count groups: %d' % count_groups)
    print('Count NA: %d [%.2f%%]' % (count_na, round(count_na * 100 / l, 2)))

    var_type = None
    if re.match('^(float|int)', str(df[c].dtype)):
        var_type = 'cont'

    vc = df[name].value_counts()
    count_most_frequent = vc.idxmax()
    print('Most frequent value %s - %d [%.2f%%]' % (count_most_frequent,
                                                    vc.max(), round(vc.max() * 100 / l, 2)))

    if var_type == 'cont':
        df[name].hist(bins='auto' if count_groups < 30 else 50)
        plt.show()

    print()


def cramers_corrected_stat(row1, row2):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    try:
        confusion_matrix = pd.crosstab(row1, row2)
    except TypeError:
        return None

    if len(confusion_matrix) == 0:
        return None

    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()

    if n == 1:
        return None

    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    d = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / d) if d > 0 else None


def calc_trusted_auc(df, c, target, n=30, default_auc=0.5):
    df_tmp = df.loc[~pd.isnull(df[c])][[c, target]]
    results = []
    if not len(df_tmp):
        return default_auc

    for i in range(n):
        sample = df_tmp.sample(frac=1, replace=True)
        try:
            results.append(roc_auc_score(sample[target], sample[c]))
        except ValueError:
            return default_auc

    return results


# r = []
# for i, c in enumerate(df_all.columns.values):
#     aucs = calc_trusted_auc(df_all, c, target)
#     if np.abs(np.mean(aucs) - 0.5) > 0.001:
#         print(i, c, round(np.mean(aucs), 4), round(np.std(aucs), 4))
#         r.append({'column': c, 'mean': round(np.mean(aucs), 4), 'std': round(np.std(aucs), 4),
#                   'min': round(np.min(aucs), 4), 'max': round(np.max(aucs), 4),
#                   'na_ratio': len(df_all.loc[~pd.isnull(df_all[c])]) / len(df_all)})
#
#
# r = pd.DataFrame(r)
# r['diff'] = np.abs(r['mean'] - 0.5)
# r.query('std < 0.07').sort_values(['diff'], ascending=[False]).to_csv('data/near_aucs_filtered.txt', index=False)
