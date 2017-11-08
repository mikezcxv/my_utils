import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Target exploration
# show_count_by_dates, show_bad_rate

# Features exploration
# show_heat_map, show_hist_by_periods, explore_categorical_to_binary_target


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


def show_bad_rate(df, date_column, target_name, file_name='bad_rate.png', folder_path='imgs',
                  sens=32, fig_size=(16, 9), inline=False):

    df_date = df[[date_column, target_name]]
    df_date[date_column] = df_date[date_column].astype("datetime64")
    df_date['dm'] = df_date[date_column].apply(lambda x: x.year * 1000 + x.month * 10 + int(x.day / sens))

    plt.clf()
    plt.subplots(figsize=(fig_size[0], fig_size[1]))
    sns.barplot(x='dm', y=target_name, data=df_date)
    # axes[ax_x, ax_y].set_title('Period: ' + str(p))

    if inline:
        plt.show()
    else:
        plt.savefig(folder_path + '/' + file_name)

    plt.clf()


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


def show_hist_by_periods(df, column_name, target_name, date_name, periods=\
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
    df_copy['ym'] = df_copy[date_name].apply(lambda x: x.year * 1000 + x.month)

    # df_raw[['circulodecredito_sameaddressapplicantcount', 'dm', 'isfraud']].groupby('dm').agg(['count', 'mean'])
    # fig, ax = plt.subplots(1,2, figsize=(10,4))
    # ax[0].hist(x, normed=True, color='grey')
    # hist, bins = np.histogram(x)
    # ax[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='grey')
    # ax[0].set_title('normed=True')
    # ax[1].set_title('hist = hist / hist.sum()')

    plt.clf()
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18))

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
