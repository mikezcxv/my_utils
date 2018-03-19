import scipy as sp
import numpy as np
import seaborn as sns


def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def analyze_confidence(data: list):
    l = []
    # for k in list_predictions.keys():
    #     a = roc_auc_score(rx.target_test, list_predictions[k])
    #     l.append(a)

    # 76.16 +- 0.68; [74.92 .. 76.95]; range: 2.03; median: 76.41
    # 75.91 +- 0.78; [74.28 .. 76.81]; range: 2.54; median: 76.16
    # 76.06 +- 0.65; [74.84 .. 77.17]; range: 2.33; median: 76.00
    # 76.06 +- 0.63; [74.79 .. 77.17]; range: 2.38; median: 76.00

    n = len(data)

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
    ci_95 = mean_confidence_interval(l, confidence=0.95)
    ci_99 = mean_confidence_interval(l, confidence=0.99)

    print('***')
    print('[%.2f .. %.2f] - 95%% mean range\'' % (ci_95[1] * 100, ci_95[2] * 100))
    print('[%.2f .. %.2f] - 99%% mean range\'' % (ci_99[1] * 100, ci_99[2] * 100))

    sns.boxplot(l)


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
