import re
import pandas as pd
import numpy as np
# import stat
from stat import *
import statsmodels.stats as stats2
import scipy.stats as stats


def find_lists_intersection(*l):
    """
    Get intersection of lists for 1 dim lists only
    :param l:
    :return:
    """
    rest = l[0]
    for i in range(1, len(l)):
        rest = list(filter(set(rest).__contains__, l[i]))

    return rest


def report_numerical(df, columns_list, folder_save='reports', file_save='numerical_variables.csv', debug=True):
    l = []
    for c in columns_list:
        l.append({'Name': c,
                  'MissingNumber': df[c].isnull().sum(),
                  'Min': np.min(df[c]),
                  'Median': np.median(df.loc[~pd.isnull(df[c])][c]),
                  'Max': np.max(df[c])
                  })

        if debug:
            print(c, df[c].isnull().sum(), np.min(df[c]), np.max(df[c]),
                  np.median(df.loc[~pd.isnull(df[c])][c]))

    d = pd.DataFrame(l)
    d.sort_values(['MissingNumber'], ascending=['True'], inplace=True)
    d = d.reindex_axis(['Name', 'MissingNumber', 'Min', 'Median', 'Max'], axis=1)

    d.to_csv(folder_save + '/' + file_save, index=False, header=True)
    return d


def report_categorical(df, columns_list, folder_save='reports', file_save='categorical_variables.csv', debug=True):
    l = []
    for c in columns_list:
        l.append({'Name': c,
                  'MissingNumber': df[c].isnull().sum(),
                  'UniqueNumber': len(df[c].value_counts())
                  })

        if debug:
            print(c, df[c].isnull().sum(), len(df[c].value_counts()))

    d = pd.DataFrame(l)
    d.sort_values(['MissingNumber'], ascending=[False], inplace=True)
    d = d.reindex_axis(['Name', 'MissingNumber', 'UniqueNumber'], axis=1)

    d.to_csv(folder_save + '/' + file_save, index=False)
    return d


def report_correlations(df, columns_list, threshold=0.95, folder_save='reports',
                        file_save='correlations.csv', debug=True):
    i, l = 0, []
    original_list = columns_list
    columns_list = find_lists_intersection(df.columns.values, columns_list)
    residual = [x for x in columns_list if x not in original_list]
    if residual:
        print(residual)

    for c1 in columns_list:
        j = 0
        for c2 in columns_list:
            if i > j:
                if re.search('^(int|float)', str(df[c1].dtype)) \
                        and re.search('^(int|float)', str(df[c2].dtype)):
                    s = stats.pearsonr(df[c1], df[c2])
                    if s[0] > threshold:
                        if debug:
                            print('P: ', s, c1, c2)
                        l.append({'Variable1': c1, 'Variable2': c2, 'Correlation': s[0]})
                else:
                    if df[c1].dtype == 'object' and df[c2].dtype == 'object':
                        s = cramers_corrected_stat(df[c1], df[c2])
                        if s is not None and s > threshold:
                            if debug:
                                print('C: ', s, c1, c2)
                            l.append({'Variable1': c1, 'Variable2': c2, 'Correlation': s})
            j += 1
        i += 1

    d = pd.DataFrame(l)
    d.sort_values(['Correlation'], ascending=[False], inplace=True)
    d = d.reindex_axis(['Variable1', 'Variable2', 'Correlation'], axis=1)

    d.to_csv(folder_save + '/' + file_save, index=False)
    return d


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
    d = d[['Name', 'Mean <=%.2f' % cutoff,
           'Mean >%.2f' % cutoff, 'Count NA',
           'Median <=%.2f' % cutoff, 'Median >%.2f' % cutoff,
           'KS stat', 'KS stat p', 't stat', 't stat p']]
    d = d.sort_values(['KS stat p', 'KS stat'], ascending=[False, True])
    return d


def common():
    show_heat_map(df, 'features_corr2', folder_path='imgs', width=24, height=11)

    show_count_by_dates_gen2(
        df_raw[[date_column, target, 'application_id']],
        date_column, pk, file_name='count_by_dates.png', folder_path='imgs',
        plt_title='# by months', fig_size=(16, 9), inline=False, xticks_position='vertical')

    show_bad_rate(df_raw[[date_column, target, pk]],
                  date_column, target, file_name='bad_rate.png', folder_path='imgs',
                  sens=32, fig_size=(16, 9), inline=False)

    df_raw[target].sum() / len(df_raw)

    floating_auc_by_dates(df_raw, ['income'], target, date_column, pk)

    for c in df:
        if c not in ['application_id', 'target']:
            df[c].hist(bins=40)
            plt.xlabel(c)
            plt.show()

    for c in only_columns:
        compare_distributions(tr, 0.5, c, inline=False, folder_save='imgs/features_info')

    features_info(df[only_columns], (0, .15, .3, .45, .6, .75, .9, 1), target, (target, pk))

    for c in df.columns.values:
        if c not in ['id', target]:
            floating_auc(df, c, target, min_section=2000)
            plt.clf()

    stats2.gof.powerdiscrepancy(oot['target'], oot['realVal'])
    stats.ks_2samp(oos_train['target'].values, oos_train['realVal'].values)
    stats.ks_2samp(oot['target'].values, oot['realVal'].values)


    report_numerical(df_raw, [x for x in df_raw.columns if (x not in ['id', target,
                # Technical or duplicated fields
                date_column, pk])
            and (df_raw[x].dtype != 'object') and (df_raw[x].dtype != 'bool')\
            # Filtered
            and (x not in skip_columns) and (x not in skip_dates)\
            and (df_raw[x].isnull().sum() < len(df_raw) - 1)
    ])

    report_correlations(df_raw, list_check)
    report_categorical(df_raw, [x for x in df_raw.columns if (x not in ['id', target,
        # Technical or duplicated fields
        date_column, pk])
        and (df_raw[x].dtype in ['object', 'bool'])\
        # Filtered
        and (x not in skip_columns) and (x not in skip_dates)\
        and (df_raw[x].isnull().sum() < len(df_raw) - 1)
    ])

