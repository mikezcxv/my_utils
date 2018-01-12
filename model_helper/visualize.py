import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import datetime
from matplotlib.dates import date2num
from model_helper.common import *


# Target exploration
# show_count_by_dates, show_bad_rate

# Features exploration
# show_heat_map, show_hist_by_periods, explore_categorical_to_binary_target

# deprecated
def show_count_by_dates(df, date_column, file_name='bad_rate.png', folder_path='imgs', fig_size=(16, 9),
                        inline=False):
    df_date = pd.DataFrame(columns=['date'])
    df_date['date'] = df[date_column]
    df_date['date'] = df_date['date'].astype("datetime64")
    df_date['date'].groupby([df_date['date'].dt.year, df_date['date'].dt.month]) \
        .count().plot(kind="bar", figsize=(fig_size[0], fig_size[1]))

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def show_count_by_dates_gen2(df, date_column, pk, file_name='bad_rate.png', folder_path='imgs',
                             plt_title='', fig_size=(10, 8), inline=False, debug=False,
                             xticks_position='horizontal'):
    df_date = pd.DataFrame(columns=['date'])
    df_date['date'] = df[date_column]
    df_date['date'] = df_date['date'].astype("datetime64")
    df_date[pk] = df[pk]

    data = []
    for d in split_by_months(df_date, 'date', pk).iterrows():
        _date, _from, _to, _count = d[0], d[1][0], d[1][1], d[1][2]
        dt = datetime.datetime(_date[0], _date[1], 1, 0, 0)
        data.append((dt, _count))

        if debug:
            print(_date, _from, _to, _count)

    fig = plt.figure(figsize=fig_size)
    plt.ylabel('count')
    plt.xlabel('dates')

    graph = fig.add_subplot(111)
    x = [date2num(date) for (date, value) in data]
    y = [value for (date, value) in data]
    graph.bar(x, y, width=17)
    graph.set_xticks(x)
    graph.set_xticklabels([date.strftime("%Y-%m") for (date, value) in data],)

    plt.xticks(rotation=xticks_position)
    plt.legend(loc="upper right")
    plt.title(plt_title)
    # graph.set_facecolor("white")

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def compare_auc(df, target_column, columns, file_name='new_comparision',
                title='Scores performance comparision'):
    fpr, tpr, _auc = [], [], []

    i = 0
    for c in columns:
        f, t, _ = roc_curve(1 - df[target_column], df[c])
        fpr.append(f), tpr.append(t)
        _auc.append(round(roc_auc_score(1 - df[target_column], df[c]), 4))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    #     ax.set_facecolor("white")

    i = 0
    for c in columns:
        plt.plot(fpr[i], tpr[i], label=c + ' (area = %0.4f)' % _auc[i])
        i += 1

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(file_name + '.png')
    plt.show()
    plt.clf()


def floating_bad_rate(df, x_column, y_column, _from, _to, count_points=100, min_in_group=30):
    fl = np.linspace(_from, _to, num=count_points)

    points_x = []
    points_y = []
    for a in fl:
        s = df.loc[(df[x_column] < a) & (df[x_column] > a - 3 * (_to - _from) / count_points)]
        l = len(s)
        if l & (l > min_in_group):
            bad_rate = s[y_column].sum() / l
            points_x.append(a)
            points_y.append(bad_rate)
            print(round(a, 4), l, round(bad_rate, 4))

    plt.plot(points_x, points_y)
    plt.show()
    plt.clf()


# def show_bad_rate(df, date_column, target_name, file_name='bad_rate.png', folder_path='imgs',
#                   sens=32, fig_size=(16, 9), inline=False):
#
#     df_date = df[[date_column, target_name]]
#     df_date[date_column] = df_date[date_column].astype("datetime64")
#     df_date['dm'] = df_date[date_column].apply(lambda x: x.year * 1000 + x.month * 10 + int(x.day / sens))
#
#     plt.clf()
#     plt.subplots(figsize=(fig_size[0], fig_size[1]))
#     sns.barplot(x='dm', y=target_name, data=df_date)
#     # axes[ax_x, ax_y].set_title('Period: ' + str(p))
#
#     if inline:
#         plt.show()
#     else:
#         plt.savefig(folder_path + '/' + file_name)
#
#     plt.clf()


def show_bad_rate(df, date_column, target_name, file_name='bad_rate.png', folder_path='imgs',
                  sens=32, fig_size=(16, 9), inline=False,
                  xticks_position='vertical', plt_title='Target rate', debug=False, pk='id'):
    df_date = df[[date_column, target_name]]
    df_date[date_column] = df_date[date_column].astype("datetime64")
    df_date['date'] = df_date[date_column].apply(lambda x: x.year * 1000 + x.month * 10 + int(x.day / sens))

    df_date['date_inner'] = df[date_column]
    df_date['date_inner'] = df_date['date_inner'].astype("datetime64")
    df_date[pk] = df[pk]

    data = []
    for d in split_by_months(df_date, 'date_inner', pk).iterrows():
        _date, _from, _to, _count = d[0], d[1][0], d[1][1], d[1][2]
        dt = datetime.datetime(_date[0], _date[1], 1, 0, 0)
        data.append((dt, _count))
        count_bad = df.loc[(df[pk] >= _from) & (df[pk] <= _to), target_name].sum()

        if debug:
            print(_date, _from, _to, _count, round(count_bad / _count, 4))

    plt.subplots(figsize=(fig_size[0], fig_size[1]))
    graph = sns.barplot(x='date', y=target_name, data=df_date)
    graph.set_xticklabels([date.strftime("%Y-%m") for (date, value) in data], )

    plt.xticks(rotation=xticks_position)
    plt.legend(loc="upper right")
    plt.title(plt_title)

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def show_target_info(df, target_c):
    l = len(df)
    count_bad = df[target_c].sum()
    print('Total: %d, bad: %d, bad rate: %.4f' % (l, count_bad, count_bad / l))


def show_heat_map(df, file_name, folder_path='imgs', width=24, height=11, inline=False, method='pearson'):
    corr_matrix = df.corr(method)
    plt.figure(figsize=(width, height))
    sns.heatmap(corr_matrix, cmap='bwr')

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def explore_categorical_to_binary_target(df, column_name, target_name, folder_path='imgs', file_prefix='bar_',
                                         periods=(0, .15, .3, .45, .6, .75, .9, 1), figsize=(18, 14),
                                         inline=False):
    print(df[column_name].value_counts())
    fp = folder_path + '/' + file_prefix + column_name + '.png'
    # TODO
    # features_info(df[[name, target_name]], periods, target, target)

    plt.subplots(figsize=figsize)
    sns.barplot(x=column_name, y=target_name, data=df)
    if inline:
        plt.show()
    else:
        plt.savefig(fp)

    plt.clf()


def show_hist_by_periods(df, column_name, target_name, date_name, periods= \
        [2016009, 2016010, 2016011, 2016012, 2017001, 2017002, 2017003],
                         bins=np.linspace(1000, 2600, num=5), avg_bad_rate=.13,
                         file_prefix='by_periods_',
                         folder_path='img_features',
                         debug=False, inline=False):
    """
    Examples of usage:
    plt_hist_by_periods(df_raw, cn=c, avg_bad_rate=.124, bins=np.linspace(0, 6, num=4), debug=False)
    plt_hist_by_periods(df_mix, cn=c, avg_bad_rate=.124, bins=[0, 0.2, 0.5, 0.8, 1, 1.5, 2, 3], debug=False)
    """
    df_copy = df[[date_name, target_name, column_name]].copy()
    if not re.search('date', str(df_copy[date_name].dtype)):
        df_copy[date_name] = df_copy[date_name].astype("datetime64")

    df_copy['ym'] = df_copy[date_name].apply(lambda x: x.year * 1000 + x.month)

    # df_raw[['circulodecredito_sameaddressapplicantcount', 'dm', 'isfraud']].groupby('dm').agg(['count', 'mean'])
    # fig, ax = plt.subplots(1,2, figsize=(10,4))
    # ax[0].hist(x, normed=True, color='grey')
    # hist, bins = np.histogram(x)
    # ax[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey')
    # ax[0].set_title('normed=True')
    # ax[1].set_title('hist = hist / hist.sum()')

    plt.clf()
    fig, axes = plt.subplots(nrows=math.ceil(len(periods) / 2), ncols=2,
                             figsize=(12, math.ceil(math.ceil(len(periods) / 2) * 4.5)))

    # [1000, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4]
    # weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    i = 0
    for p in periods:
        ax_x, ax_y = int(i / 2), int(i % 2)

        # TODO revert NA
        #     ss[[cn, 'isfraud']].hist(ax=axes[ax_x, ax_y], column=cn, by='isfraud', bins=12)
        #     ss.loc[pd.isnull(ss[cn]), cn] = 3.1
        ss = df_copy.loc[np.where(df_copy['ym'] == p)]
        ss = ss.loc[~pd.isnull(ss[column_name])]

        period_bad_rate = len(ss.loc[ss[target_name] == 1]) / len(ss)
        if debug:
            print(ss[[target_name, column_name]].groupby(column_name).agg(['count', 'mean']))

        ss['bin'] = pd.cut(ss[column_name], bins, labels=False)

        #     b = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        #     axes[ax_x, ax_y].hist(ss.loc[ss.isfraud == 1, 'binned'], bins=b, alpha=0.9, color="black")
        #     axes[ax_x, ax_y].hist(ss.loc[ss.isfraud == 0, 'binned'], bins=b, alpha=0.6)
        #     plt.legend(loc='upper right')
        sns.barplot(x='bin', y=target_name, data=ss, ax=axes[ax_x, ax_y])
        axes[ax_x, ax_y].hlines(avg_bad_rate, -1, len(ss['bin']), linestyle='--', linewidth=1)
        axes[ax_x, ax_y].hlines(period_bad_rate, -1, len(ss['bin']), linestyle='solid', linewidth=1)
        if debug:
            print(i)

        axes[ax_x, ax_y].set_title('Period: ' + str(p))
        axes[ax_x, ax_y].set_ylim(0, 1.2)
        axes[ax_x, ax_y].set_xlim(-1, len(bins) - 1)

        i += 1

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_prefix + column_name + '.png')

    plt.clf()

    # ax[0].hist(df_raw.loc[np.where(df_raw['dm'] == 2016009)][[cn]], normed=True, color='grey')
    # ax[1].hist(df_raw.loc[np.where(df_raw['dm'] == 2016010)][[cn]], normed=True, color='grey')

    # g = sns.FacetGrid(df_raw[['dm', 'isfraud', cn]], col="dm", size=4, aspect=.5)
    # g.map(plt.hist, "isfraud", cn);


def show_hist_by_periods_gen2(df, c, target, date_column, periods= \
        [2016009, 2016010, 2016011, 2016012, 2017001, 2017002, 2017003], \
                              num_bins=5, file_prefix='by_periods_', folder_path='img_features', debug=False,
                              inline=False):
    _min, _max = np.min(df[c]), np.max(df[c])
    bad_rate = round(df[target].sum() / len(df), 3)
    bins = np.linspace(_min, _max, num=num_bins)
    show_hist_by_periods(df, c, target, date_column, periods=periods, bins=bins, avg_bad_rate=bad_rate,
                         file_prefix=file_prefix, folder_path=folder_path, debug=debug, inline=inline)


def show_bins_by_periods(df, column_name, target_name, date_name, periods= \
        [2016009, 2016010, 2016011, 2016012, 2017001, 2017002, 2017003],
                         num_bins=5, file_prefix='by_periods_', folder_path='img_features', debug=False, inline=False):
    bad_rate = round(df[target_name].sum() / len(df), 3)

    df_copy = df[[date_name, target_name, column_name]].copy()
    if not re.search('date', str(df_copy[date_name].dtype)):
        df_copy[date_name] = df_copy[date_name].astype("datetime64")

    df_copy['ym'] = df_copy[date_name].apply(lambda x: x.year * 1000 + x.month)

    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))

    info = pd.qcut(df_copy[column_name], num_bins, duplicates='drop').value_counts()
    num_bins = min(num_bins, len(info))

    i, bad_rates = 0, []

    for p in periods:
        df_bin_by_period = df_copy.loc[np.where(df_copy['ym'] == p)]

        j = 0
        for v in info.index.categories.values:
            if i == 0 & j == 0:
                bad_rates.append([])

            df_bin = df_bin_by_period.loc[~pd.isnull(df_bin_by_period[column_name])
                                          & (df_bin_by_period[column_name] > v.left) & (
                                          df_bin_by_period[column_name] <= v.right)]

            bin_size = len(df_bin)
            period_bad_rate = df_bin[target_name].sum() / bin_size if bin_size else 0

            bad_rates[j].append(period_bad_rate)
            j += 1

        i += 1

    ax.hlines(bad_rate, -1, len(periods), linestyle='--', linewidth=1, label='Average bad rate')

    for i in range(0, num_bins):
        plt.plot(bad_rates[i], label=info.index.categories[i])

    plt.legend(loc='upper right')
    plt.title('bad rate by equal bins for feature %s' % column_name)
    plt.ylabel('bad rate')
    ax.set_xticks(range(0, len(periods)))
    ax.set_xticklabels(
        ['%02d/%d' % (p % 100, int(p / 1000)) for p in periods])

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_prefix + column_name + '.png')

    plt.clf()


def compare_distributions(df, split_size, column_name, labels=['Time frame #1', 'Time frame #2'],
                          folder_save='imgs', inline=False):
    a = df.iloc[:math.ceil(len(df) * split_size), :][[column_name]]
    b = df.iloc[math.ceil(len(df) * split_size):, :][[column_name]]

    a = a.loc[~pd.isnull(a[column_name])][column_name]
    b = b.loc[~pd.isnull(b[column_name])][column_name]

    plt.hist([a, b], alpha=0.7, label=labels)

    plt.ylabel('count')
    plt.legend(loc='upper right')
    plt.title(column_name)
    if inline:
        plt.show()
    else:
        plt.savefig(folder_save + '/info_' + column_name + '_count.png')

    plt.clf()

    fig, ax = plt.subplots()
    ax.hist(a, 30, histtype='step', alpha=0.8, normed=True, label=labels[0])
    ax.hist(b, 30, histtype='step', alpha=0.8, normed=True, label=labels[1])
    ax.legend(loc='upper left')
    plt.ylabel('density')
    plt.title(column_name)
    if inline:
        plt.show()
    else:
        plt.savefig(folder_save + '/info_' + column_name + '_density.png')

    plt.clf()

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.boxplot([a, b], vert=True, patch_artist=False)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len([a, b]))], )
    ax.set_ylabel(column_name)
    plt.title(column_name)

    plt.setp(ax, xticks=[y + 1 for y in range(len([a, b]))], xticklabels=labels)
    if inline:
        plt.show()
    else:
        plt.savefig(folder_save + '/info_' + column_name + '_boxplot.png')

    plt.clf()


def show_binary_hist(df, column_name, target_name, file_name, folder_path, inline=False):
    _df = df.copy()
    _df = _df[~pd.isnull(_df[column_name])]

    plt.hist(_df.loc[_df[target_name] == 1, column_name], 200, alpha=0.8, label='Bad', color="red")
    plt.hist(_df.loc[_df[target_name] == 0, column_name], 200, alpha=0.5, label='Good')

    plt.legend(loc='upper right')

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def draw_distributions(result, target_column, file_name, folder_path, inline=False):
    plt.hist(result.loc[result.realVal == 1, target_column], 100, alpha=0.8, label='Bad', color="red")
    plt.hist(result.loc[result.realVal == 0, target_column], 100, alpha=0.5, label='Good')
    plt.legend(loc='upper right')

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def draw_distributions_gen2(result, target_column, file_name, folder_path, inline=False, title=None,
                      cumulative=False, normed=1, figsize=(12, 8)):
    plt.clf()
    plt.figure(figsize=figsize)
    plt.hist(result.loc[result.realVal == 1, target_column], 100, alpha=0.8, label='Bad', color="red",
             normed=normed, cumulative=cumulative)
    plt.hist(result.loc[result.realVal == 0, target_column], 100, alpha=0.5, label='Good',
             normed=normed, cumulative=cumulative)
    plt.legend(loc='upper right')
    plt.ylabel('Density' if not cumulative else '')
    if title:
        plt.title(title)

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + ('density_' if not cumulative else '') + file_name)

    plt.clf()


def draw_distributions_2_samples(a, b, score_column, file_name, folder_path, figsize=(12, 8), inline=False, title=None,
                      cumulative=False, normed=1, labels=['Bad', 'Good']):
    plt.clf()
    a = a.loc[~pd.isnull(a[score_column])]
    b = b.loc[~pd.isnull(b[score_column])]
    plt.figure(figsize=figsize)
    plt.hist(a[score_column], 100, label=labels[0], color="red", histtype='step', normed=normed, cumulative=cumulative)
    plt.hist(b[score_column], 100, label=labels[1], histtype='step', normed=normed, cumulative=cumulative)
    plt.legend(loc='upper right')
    plt.ylabel('Density' if not cumulative else '')
    plt.title(title)

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + ('density_' if not cumulative else '') + file_name)

    plt.clf()


def get_populations_diff(df, cutoff=None, skip_columns=[], target='target',
                         get_percent_bad=None):
    l = []

    df = df.sort_values(target, ascending=[True])
    len_df = len(df)

    for c in df.columns.values:
        if c not in skip_columns:
            if get_percent_bad:
                a = df.iloc[:math.ceil(len_df * (1 - get_percent_bad)), :]
                b = df.iloc[math.ceil(len_df * (1 - get_percent_bad)):, :]
                cutoff = 1 - get_percent_bad
            else:
                a = df.loc[df[target] <= cutoff]
                b = df.loc[df[target] > cutoff]

            st = stats1.ks_2samp(a[c], b[c])
            st_t = stats1.ttest_ind(a.loc[~pd.isnull(a[c])][c],
                                    b.loc[~pd.isnull(b[c])][c])
            l.append({
                'Name': c,
                'Mean <=%.2f' % cutoff: round(np.mean(a[c]), 4),
                'Mean >%.2f' % cutoff: round(np.mean(b[c]), 4),
                'Median <=%.2f' % cutoff: round(np.median(a[c]), 4),
                'Median >%.2f' % cutoff: round(np.median(b[c]), 4),
                'KS stat': round(st[0], 4),
                'KS stat p': round(st[1], 4),
                't stat': round(st_t[0], 4),
                't stat p': round(st_t[1], 4),
                'Count NA': df[c].isnull().sum()
            })

    d = pd.DataFrame(l)
    d = d[['Name', 'Mean <=%.2f' % cutoff, \
           'Mean >%.2f' % cutoff, 'Count NA',
           'Median <=%.2f' % cutoff, 'Median >%.2f' % cutoff,
           'KS stat', 'KS stat p', 't stat', 't stat p']]
    d = d.sort_values(['KS stat p', 'KS stat'], ascending=[False, True])
    return d


def get_populations_diff_with_plots(df, cutoff=None, skip_columns=[], target='target',
                                    get_percent_bad=None):
    l = []

    df = df.sort_values(target, ascending=[True])
    len_df = len(df)

    for c in df.columns.values:
        if c not in skip_columns:
            if get_percent_bad:
                a = df.iloc[:math.ceil(len_df * (1 - get_percent_bad)), :]
                b = df.iloc[math.ceil(len_df * (1 - get_percent_bad)):, :]
                cutoff = 1 - get_percent_bad
            else:
                a = df.loc[df[target] <= cutoff]
                b = df.loc[df[target] > cutoff]

            draw_distributions_2_samples(b, a, c, 'scores_comparison%s.png' % c,
                                         'reports', inline=False,
                                         title=c, cumulative=False, normed=False, labels=['10% of the worst scores',
                                                                                          '90% of the best scores'])

            st = stats1.ks_2samp(a[c], b[c])
            st_t = stats1.ttest_ind(a.loc[~pd.isnull(a[c])][c],
                                    b.loc[~pd.isnull(b[c])][c])
            l.append({
                'Name': c,
                'Mean <=%.2f' % cutoff: round(np.mean(a[c]), 4),
                'Mean >%.2f' % cutoff: round(np.mean(b[c]), 4),
                'Median <=%.2f' % cutoff: round(np.median(a[c]), 4),
                'Median >%.2f' % cutoff: round(np.median(b[c]), 4),
                'KS stat': round(st[0], 4),
                'KS stat p': round(st[1], 4),
                't stat': round(st_t[0], 4),
                't stat p': round(st_t[1], 4),
                'Count NA': df[c].isnull().sum()
            })

    d = pd.DataFrame(l)
    d = d[['Name', 'Mean <=%.2f' % cutoff, \
           'Mean >%.2f' % cutoff, 'Count NA',
           'Median <=%.2f' % cutoff, 'Median >%.2f' % cutoff,
           'KS stat', 'KS stat p', 't stat', 't stat p']]
    d = d.sort_values(['KS stat p', 'KS stat'], ascending=[False, True])
    return d


def draw_roc_curve(y_pred, target_test, file_name, folder_path, avg_auc_num, inline=False):
    fpr, tpr, _ = roc_curve(target_test, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % avg_auc_num)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


def floating_auc(df, column_name, target_name, min_section=500):
    l = len(df)
    c, target = column_name, target_name
    split_ranges = np.linspace(0, 1, num=round(l / max(1, min_section)))
    count_ranges = len(split_ranges) - 1
    hist = []
    skip_count = 0
    for i in range(0, count_ranges):
        # print("From %.3f to %.3f" % (int(split_ranges[i] * l), int(split_ranges[i + 1] * l)))
        a = df.iloc[int(split_ranges[i] * l): int(split_ranges[i + 1] * l), :]
        try:
            hist.append(roc_auc_score(a.loc[~pd.isnull(a[c])][target], a.loc[~pd.isnull(a[c])][c]))
        except ValueError:
            skip_count += 1

    x = range(0, len(hist))
    plt.plot(range(0, len(hist)), hist)
    m, b = np.polyfit(x, hist, 1)

    if skip_count:
        print('Skip: %d' % skip_count)

    if hist:
        count_na = len(df.loc[pd.isnull(df[c])])
        plt.text(0, max(hist) - ((max(hist) - min(hist)) / 80), 'Mean auc: %.4f' % (np.mean(hist)))
        plt.text(0, max(hist) - ((max(hist) - min(hist)) / 10 + 0.01),
                 'Count NA: %d [%.2f%%]' % (count_na, (count_na / l) * 100))
        plt.ylabel('AUC')
        plt.xlabel(column_name)
        plt.plot(x, m * x + b, ':')
        plt.plot(x, [0.5] * len(hist), '--')

        plt.show()
        plt.clf()
        plt.hist(hist, bins=30)
        plt.clf()
    else:
        print('Nothing to do..')


def floating_auc_by_dates(df, columns, target_name, date_column, pk,
                          plt_title='', debug=False):
    """
    floating_auc_by_dates(df_cleaned_vs_date, ['zibby_score', 'CSFraud_Score', 'FICECLV9_SCORE', 'FICO_plus_zibbi'],
                      'SecondPaymentDelinquencyClass_r', 'created_at', 'id',
                      'AUC for different periods', True)
    """
    l = len(df)
    target = target_name

    min_in_month, skip_count = 30, 0
    hist = []
    dates = []
    data = {}
    for d in split_by_months(df, date_column, pk).iterrows():
        _date, _from, _to, _count = d[0], d[1][0], d[1][1], d[1][2]
        if _count < min_in_month:
            if debug:
                print('Skip month ', _date, 'with %d' % _count)
            skip_count += 1
            continue

        if debug:
            print(d[0], '---', d[1][0], d[1][1], d[1][2])

        a = df.where((df[pk] >= _from) & (df[pk] <= _to))

        try:
            dt = datetime.datetime(_date[0], _date[1], 1, 0, 0)
            # hist.append(sc), dates.append(dt)
            for c in columns:
                sc = roc_auc_score(a.loc[~pd.isnull(a[c])][target], a.loc[~pd.isnull(a[c])][c])
                if c in data:
                    data[c].append((dt, sc))
                else:
                    data[c] = [(dt, sc)]

        except ValueError:
            skip_count += 1

    count_ranges = len(hist)

    #     x = range(0, len(hist))
    #     plt.plot(range(0, len(hist)), hist)
    #     m, b = np.polyfit(x, hist, 1)

    if skip_count:
        print('Skip: %d' % skip_count)

    if data:
        fig = plt.figure(figsize=(10, 8))
        plt.ylabel('AUC')
        plt.xlabel('Dates')

        graph = fig.add_subplot(111)
        i = 0
        for k, v in data.items():
            x = [date2num(date) for (date, value) in v]
            y = [value for (date, value) in v]
            graph.plot(x, y, '-o', label=k)
            if i == 0:
                graph.set_xticks(x)
                graph.set_xticklabels([date.strftime("%Y-%m") for (date, value) in v])

        plt.legend(loc="upper right")
        plt.title(plt_title)
        plt.show()

    else:
        print('Nothing to do..')


# TODO check
# @deprecated
def draw_barplots(df, target_name, period="year", file_suffix=""):
    # sns.factorplot("kind", "pulse", "diet", exercise, kind="point", size=4, aspect=2)
    for c in df.columns.values:
        if c not in {'id', target_name}:
            count_groups = len(df[c].value_counts().to_dict())
            if count_groups > 50 or count_groups < 2:
                print("Skip %s; Num classes %d" % (c, count_groups))
                continue

            _iv = iv2(df, c, target_name)
            plt.figure(figsize=(18, 8))
            ax = sns.barplot(x=period, y=target_name, hue=c, data=df)
            ax.set(xlabel=c + ". NA: " + str(len(df.loc[np.isnan(df[c])])) + "; IV: " + str(_iv))
            plt.savefig("imgs/" + file_suffix + "_" +  c + ".png")
            plt.clf()


#Draft
def plot_one_xgboost_tree():
    # plot_tree(model, 0)

    fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(50, 50)
    fig.set_figwidth(30)
    fig.set_figheight(30)
    plt.show()


# Draft
# from sklearn.metrics import accuracy_score
#
#
# def calc_cumulative_gains(df: pd.DataFrame, actual_col: str, predicted_col: str, probability_col: str):
#     df.sort_values(by=probability_col, ascending=False, inplace=True)
#
#     subset = df[df[predicted_col] == True]
#
#     rows = []
#     for group in np.array_split(subset, 10):
#         score = accuracy_score(group[actual_col].tolist(),
#                                group[predicted_col].tolist(),
#                                normalize=False)
#
#         rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})
#
#     lift = pd.DataFrame(rows)
#
#     # Cumulative Gains Calculation
#     lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
#     lift['PercentCorrect'] = lift.apply(
#         lambda x: (100 / (lift['NumCorrectPredictions'].sum() + 0.00001)) * x['RunningCorrect'], axis=1)
#     lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()
#     lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
#         lambda x: 100 if (100 / (lift['NumCorrectPredictions'].sum() + 0.00001)) * x > 100 else (100 / (lift[
#                                                                                                             'NumCorrectPredictions'].sum() + 1)) * x)
#     lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
#     lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
#     lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
#         lambda x: (100 / (lift['NumCorrectPredictions'].sum() + 0.0001) * x))
#
#     # Lift Chart
#     lift['NormalisedPercentAvg'] = 1
#     lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / lift['PercentAvgCase']
#
#     return lift
#
#
# def plot_cumulative_gains(lift: pd.DataFrame):
#     fig, ax = plt.subplots()
#     fig.canvas.draw()
#
#     handles = []
#     handles.append(ax.plot(lift['PercentCorrect'], 'r-', label='Percent Correct Predictions'))
#     handles.append(ax.plot(lift['PercentCorrectBestCase'], 'g-', label='Best Case (for current model)'))
#     handles.append(ax.plot(lift['PercentAvgCase'], 'b-', label='Average Case (for current model)'))
#     ax.set_xlabel('Total Population (%)')
#     ax.set_ylabel('Number of Respondents (%)')
#
#     ax.set_xlim([0, 9])
#     ax.set_ylim([10, 100])
#
#     labels = [int((label + 1) * 10) for label in [float(item.get_text()) for item in ax.get_xticklabels()]]
#
#     ax.set_xticklabels(labels)
#
#     fig.legend(handles, labels=[h[0].get_label() for h in handles])
#     fig.show()
#
#
# def plot_lift_chart(lift: pd.DataFrame):
#     plt.figure()
#     plt.plot(lift['NormalisedPercentAvg'], 'r-', label='Normalised \'response rate\' with no model')
#     plt.plot(lift['NormalisedPercentWithModel'], 'g-', label='Normalised \'response rate\' with using model')
#     plt.legend()
#     plt.show()
#
#
# plot_cumulative_gains(calc_cumulative_gains(oot, 'realVal', target, target))

# TODO Draft calibration plot
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
#                              f1_score)
# from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# from sklearn.model_selection import train_test_split
#
#
# # Create dataset of classification task with many redundant and few
# # informative features
# X, y = datasets.make_classification(n_samples=100000, n_features=20,
#                                     n_informative=2, n_redundant=10,
#                                     random_state=42)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
#                                                     random_state=42)
#
#
# def plot_calibration_curve(est, name, fig_index):
#     """Plot calibration curve for est w/o and with calibration. """
#     # Calibrated with isotonic calibration
#     isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')
#
#     # Calibrated with sigmoid calibration
#     sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')
#
#     # Logistic regression with no calibration as baseline
#     lr = LogisticRegression(C=1., solver='lbfgs')
#
#     fig = plt.figure(fig_index, figsize=(10, 10))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 1), (2, 0))
#
#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#     for clf, name in [(lr, 'Logistic'),
#                       (est, name),
#                       (isotonic, name + ' + Isotonic'),
#                       (sigmoid, name + ' + Sigmoid')]:
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         if hasattr(clf, "predict_proba"):
#             prob_pos = clf.predict_proba(X_test)[:, 1]
#         else:  # use decision function
#             prob_pos = clf.decision_function(X_test)
#             prob_pos = \
#                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
#
#         clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
#         print("%s:" % name)
#         print("\tBrier: %1.3f" % (clf_score))
#         print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
#         print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
#         print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
#
#         fraction_of_positives, mean_predicted_value = \
#             calibration_curve(y_test, prob_pos, n_bins=10)
#
#         ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#                  label="%s (%1.3f)" % (name, clf_score))
#
#         ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
#                  histtype="step", lw=2)
#
#     ax1.set_ylabel("Fraction of positives")
#     ax1.set_ylim([-0.05, 1.05])
#     ax1.legend(loc="lower right")
#     ax1.set_title('Calibration plots  (reliability curve)')
#
#     ax2.set_xlabel("Mean predicted value")
#     ax2.set_ylabel("Count")
#     ax2.legend(loc="upper center", ncol=2)
#
#     plt.tight_layout()
#
# # Plot calibration curve for Gaussian Naive Bayes
# plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)
#
# # Plot calibration curve for Linear SVC
# plot_calibration_curve(LinearSVC(), "SVC", 2)
#
# plt.show()
