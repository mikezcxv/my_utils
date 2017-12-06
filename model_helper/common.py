import re
import os
import sys
import numpy as np
import operator


# sys.path.append(os.path.expanduser('~/PyProjects/my_utils'))
# from model_helper.common import *

def split_by_months(df, date_column, range_column=None):
    """
    The way to use it
    for d in split_by_months(df_cleaned_vs_date, 'created_at', 'id').iterrows():
        print(d[0], '---', d[1][0], d[1][1], d[1][2])
    """
    if not range_column:
        range_column = date_column

    return df[range_column].groupby([df[date_column].dt.year, df[date_column].dt.month]) \
        .agg(['min', 'max', 'count'])


# TODO create separate helper
def report_useless_features(model, threshold_useless=3, debug=False):
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

    return list_useless, almost_useless


# DF related
def shape_info(df, msg, prefix='', debug=True):
    if debug:
        prefix = prefix + ' ' if prefix else ''
        print(prefix + msg + ': %s x %s' % df.shape)


def find_columns_like(df, pattern, debug=True):
    res = []
    for c in df.columns.values:
        if re.search(pattern, c):
            if debug:
                print("'" + c + "', ")
            res.append(c)

    return res


def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if (np.max(df[col]) <= 127) and (np.min(df[col] >= -128)):
            df[col] = df[col].astype(np.int8)
        elif (np.max(df[col]) <= 32767) and (np.min(df[col] >= -32768)):
            df[col] = df[col].astype(np.int16)
        elif (np.max(df[col]) <= 2147483647) and (np.min(df[col] >= -2147483648)):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)


def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)


def get_pairs(columns):
    result = []
    for i in range(0, len(columns)):
        for j in range(0, len(columns)):
            if j > i:
                result.append(columns[i] + '*' + columns[j])

    return result


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


def asd():
    sys.exit()


# Dictionary related
def set_if_none(data, field, value):
    if data[field] is None:
        data[field] = value
    return data


def sort_dictionary(data, reverse=True):
    return sorted(data.items(), key=operator.itemgetter(1), reverse=reverse)


# File system helper
# TODO expand helper
def get_dir_files(dir_path):
    res = []
    for dirname, dirnames, filenames in os.walk(dir_path):
        # for subdirname in dirnames:
        #      print(os.path.join(dirname, subdirname))

        for filename in filenames:
            res.append(os.path.join(dirname, filename))

            #     if '.git' in dirnames:
            # don't go into any .git directories.
            #         dirnames.remove('.git')

    return res


# TODO analyze NA
# Trying to find custom designations of NA
# j = 0
# total_len = len(df_all)
# for c in df_all.columns.values:
#     different_groups = df_all[c].unique()
#     if len(different_groups) > 1 and (
#             len(find_lists_intersection(['-1', -1, -999, 999], different_groups))
#     ):
#
#         count_weidr = len(df_all.loc[df_all[c].isin(['-1', -1, -999, 999])])
#         if count_weidr / total_len > 0.01:
#             print(c, count_weidr)
#
#     j += 1
#     if j > 500:
#         break