import pandas as pd
from model_helper.common import *
from model_helper.stat import *


# TODO remove as it is in a common helper
def shape_info(df, msg, prefix='After', debug=True):
    if debug:
        print(prefix + ' ' + msg + ': %s x %s' % df.shape)


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, val_series=[], tst_series=[], target=None,
                  min_samples_leaf=20, smoothing=1, noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), \
           add_noise(ft_tst_series, noise_level)


def default_clean_up(df, target_column, main_date_column, dont_convert=(), dates_columns=(), debug=True,
                     min_in_column=30, drop_univariate=False):
    df = df.loc[~pd.isnull(df[target_column])]
    df.dropna(axis=1, how='all', thresh=min_in_column, subset=None, inplace=True)
    shape_info(df, 'drop NA with threshold ' + str(min_in_column), debug=debug)

    df = dates_to_age(df, dates_columns, main_date_column)
    shape_info(df, 'dates conver', debug=debug)

    df = handle_cat(df, dont_convert, debug=True)
    shape_info(df, 'cats handle', debug=debug)

    if drop_univariate:
        univariance_columns = find_low_variance(df, 0, False)
        df.drop(univariance_columns, axis=1, inplace=True)
        shape_info(df, 'drop univariance', debug=debug)

    return df


def handle_cat(df, dont_convert=('created_at', 'ssn'), debug=False):
    df_object = df.select_dtypes(include=['object']).copy()
    columns_to_convert = df_object.columns
    columns_to_convert = [x for x in columns_to_convert if x not in dont_convert]
    df_object = df_object[columns_to_convert]
    df_object[columns_to_convert] = df_object.apply(lambda x: x.str.strip())

    if debug:
        for c in columns_to_convert:
            print('%d %s' % (len(df[c].value_counts().to_dict()), c))

    return pd.get_dummies(df, columns=columns_to_convert, dummy_na=True, drop_first=True)


# Draft func
def handle_cat_encode(df, target_column, dont_convert=('created_at', 'ssn'), debug=False,
                      min_samples_leaf=20, smoothing=1, noise_level=0):
    df_object = df.select_dtypes(include=['object']).copy()
    columns_to_convert = df_object.columns
    columns_to_convert = [x for x in columns_to_convert if x not in dont_convert]
    df_object = df_object[columns_to_convert]
    df_object[columns_to_convert] = df_object.apply(lambda x: x.str.strip())

    if debug:
        for c in columns_to_convert:
            print('%d %s' % (len(df[c].value_counts().to_dict()), c))
            df[c], a, b = target_encode(df[c], df[:10][c], df[:10][c], target_column,
                                        min_samples_leaf=min_samples_leaf,
                                        smoothing=smoothing, noise_level=noise_level)

    return df


def dates_to_age(df, dates_columns, compare_date_name, drop_original=True, debug=False, with_utc=True):
    if not dates_columns:
        return df

    drop = find_lists_intersection(dates_columns, df.columns.values)
    for c in drop:
        if with_utc:
            df['converted_' + c] = pd.to_datetime(df[compare_date_name], utc=True) - pd.to_datetime(df[c], errors='coerce', utc=True)
        else:
            df['converted_' + c] = pd.to_datetime(df[compare_date_name]) - pd.to_datetime(df[c], errors='coerce')

        df['converted_' + c] = df['converted_' + c].map(lambda x: x.days)

        if debug:
            print(df['converted_' + c].describe())

    if drop_original:
        df.drop(drop, axis=1, inplace=True)

    return df


def drop_obj(df):
    df_obj = df.select_dtypes(include=['object']).copy()
    for c in df_obj:
        df.drop([c], axis=1, inplace=True)


# TODO check
def simple_cast_object(df):
    # Deal with categorical values
    df_numeric = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    return pd.concat([df_numeric, df_obj], axis=1)