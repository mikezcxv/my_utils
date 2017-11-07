import re
import sys
import numpy as np


# sys.path.append(os.path.expanduser('~/PyProjects/my_utils'))
# from model_helper.common import *


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


def find_columns_like(df, pattern, debug=True):
    res = []
    for c in df.columns.values:
        if re.search(pattern, c):
            if debug:
                print("'" + c + "', ")
            res.append(c)

    return res


def asd():
    sys.exit()


def get_pairs(columns):
    result = []
    for i in range(0, len(columns)):
        for j in range(0, len(columns)):
            if j > i:
                result.append(columns[i] + '*' + columns[j])

    return result


def set_if_none(data, field, value):
    if data[field] is None:
        data[field] = value
    return data


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

