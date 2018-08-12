import pandas as pd
import scipy as sp
import numpy as np
import seaborn as sns
import statsmodels


def get_binomial_ci(cnt_positive, cnt_total, ci=0.95, pr=3, simple=True, min_total=30, show_plt=False):
    a = statsmodels.stats.proportion.proportion_confint(cnt_positive, cnt_total, alpha=1 - ci, method='wilson')
    ratio = cnt_positive / cnt_total

    if simple:
        b = round(ratio, pr)
    else:
        b = str(round(ratio, pr)) + ' : %d from %d' % (cnt_positive, cnt_total)

    if cnt_total < min_total:
        return np.NaN, np.NaN
    else:
        #         if show_plt:
        #             er = [[ratio - float(a[0])], [float(a[1]) - ratio]]
        #             plt.errorbar(x=np.arange([1]), y=ratio, yerr=er, fmt='ro--',
        #                          linewidth=1, ecolor=colors[i], capsize=3, capthick=1)
        #             plt.show()

        return str(round(a[0], pr)) + ' .. ' + str(round(a[1], pr)), b


def get_normal_ci(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def analyze_confidence(data: list):
    l = data
    # for k in list_predictions.keys():
    #     a = roc_auc_score(rx.target_test, list_predictions[k])
    #     l.append(a)

    # 76.16 +- 0.68; [74.92 .. 76.95]; range: 2.03; median: 76.41
    # 75.91 +- 0.78; [74.28 .. 76.81]; range: 2.54; median: 76.16
    # 76.06 +- 0.65; [74.84 .. 77.17]; range: 2.33; median: 76.00
    # 76.06 +- 0.63; [74.79 .. 77.17]; range: 2.38; median: 76.00

    n = len(l)

    s, m = np.std(l), np.mean(l)
    print('Mean: %.3f; Std: %.5f; Range: %.5f' % (m * 100, s * 100, round(max(l) - min(l), 7)))

    m *= 100
    s *= 100

    print('***')
    print('[%.2f .. %.2f] - Real range' % (min(l) * 100, max(l) * 100))
    print('[%.2f .. %.2f] - 95%%' % (m - 1.96 * s, m + 1.96 * s))
    print('[%.2f .. %.2f] - 99%% ' % (m - 2.56 * s, m + 2.56 * s))

    print('[%.2f .. %.2f] - 95%% mean range' % (m - 2.56 * s / math.sqrt(n), m + 2.56 * s / math.sqrt(n)))
    print('[%.2f .. %.2f] - 99%% mean range' % (m - 3 * s / math.sqrt(n), m + 3 * s / math.sqrt(n)))
    ci_95 = get_normal_ci(l, confidence=0.95)
    ci_99 = get_normal_ci(l, confidence=0.99)

    print('***')
    print('[%.2f .. %.2f] - 95%% mean range\'' % (ci_95[1] * 100, ci_95[2] * 100))
    print('[%.2f .. %.2f] - 99%% mean range\'' % (ci_99[1] * 100, ci_99[2] * 100))

    sns.boxplot(l)


pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def get_ci(row, ci=0.95, pr=3, simple=True, tr=30):
    a = statsmodels.stats.proportion.proportion_confint(row['clc_id'], row['imp_id'], alpha=1 - ci, method='wilson')
    b = round(row['clc_id'] / row['imp_id'], pr) if simple else str(
        round(row['clc_id'] / row['imp_id'], pr)) + ' for %d impressions' % row['imp_id']
    if row['imp_id'] < tr:
        return (np.NaN, np.NaN)
    else:
        return (str(round(a[0], pr)) + ' .. ' + str(round(a[1], pr)), b)


# # df.loc[df.brand == 'RACQ']['ranking'].value_counts()
# info = df[['brand', 'ranking', 'imp_id', 'clc_id']]\
#     .loc[df.brand.isin(comparison_brands)]\
#     .groupby(('ranking', 'brand')).agg('count')\
#     .apply(lambda row: get_binomial_ci(row['clc_id'], row['imp_id'], ci=0.95, simple=True), axis=1)


def plot_error_bar(info, n_brands=4, n_ranks=5):
    # plt.plot(info['clc_id'][:n_brands * n_ranks].reshape((-1, n_brands)))

    # er = [[0.002, 0.009], [0.039, 0.047]]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # ax.set_facecolor((1.0, 1.0, 1.0))
    fig.set_facecolor((1.0, 1.0, 1.0))

    colors = ['k', 'r', 'g', 'b', 'y', 'b', 'c', 'm', 'k']

    labels = []
    for i in range(n_brands):
        l = info['imp_id'].values[:n_brands * n_ranks].reshape((-1, n_brands))[:, i]
        labels.append(info.index.get_level_values(1)[i])
        bottom, upper = [], []
        for q in l:
            it = q.split(' .. ')
            bottom.append(float(it[0]))
            upper.append(float(it[1]))

        y = info['clc_id'][:n_brands * n_ranks].reshape((-1, n_brands))[:, i]
        er = [y - bottom, upper - y]

        ax.errorbar(x=np.linspace(1, n_ranks, n_ranks) + i / 20, y=y,
                    yerr=er, fmt=colors[i] + 'o--', linewidth=1, ecolor=colors[i], capsize=3, capthick=1)

    plt.title(r'$\bf%s$ CTR comparison' % vertical.replace(' ', '\ '))
    plt.legend(labels)
    plt.xlabel('Item ranking')


# from math import sqrt
#
# def confidence(clicks, impressions):
#     n = impressions
#     if n == 0: return 0
#     z = 1.96 #1.96 -> 95% confidence
#     phat = float(clicks) / n
#     denorm = 1. + (z*z/n)
#     enum1 = phat + z*z/(2*n)
#     enum2 = z * sqrt(phat*(1-phat)/n + z*z/(4*n*n))
#     return (enum1-enum2)/denorm, (enum1+enum2)/denorm
#
# def wilson(clicks, impressions):
#     if impressions == 0:
#         return 0
#     else:
#         return confidence(clicks, impressions)


# Draft for unbalanced

# from sklearn.metrics import confusion_matrix, log_loss, f1_score, accuracy_score, average_precision_score,\
#     brier_score_loss, classification_report, cohen_kappa_score
#
# # r_prev = r.copy()
# r = result.copy()
#
# print('Log loss:', round(log_loss(r['realVal'], r['target']), 5))
# print('ROC AUC:', round(roc_auc_score(r['realVal'], r['target']), 5))
#
# for i, cat_off in enumerate([10, 40, 42, 45, 50, 55, 57, 60, 65]):
#     name = 'ct_%s' % str(cat_off)
#     r[name] = np.where(r['target'] > cat_off / 100, 1, 0)
#     print('\n' + ('\t'.join(['Cat OFF', 'F1', 'F1w', 'Acc', 'APR', 'Bri', 'Cappa'])) +
#           '\n' + ('%d\t' + '\t'.join((['%.3f'] * 6))) % (
#             cat_off,
#             f1_score(r['realVal'], r[name]),
#             f1_score(r['realVal'], r[name], average='weighted'),
#             accuracy_score(r['realVal'], r[name]),
#             average_precision_score(r['realVal'], r[name]),
#             brier_score_loss(r['realVal'], r[name]),
#             cohen_kappa_score(r['realVal'], r[name])
#         )
#     )
# #     print(classification_report(r['realVal'], r[name],
# #                                           target_names=['Not clicked', 'Clicked']))
#     print('Confusion matrix\n', confusion_matrix(r[name], r['realVal']))
