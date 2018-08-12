

def show_cat_counts(df):
    df_object = df.select_dtypes(include=['object']).copy()
    columns_to_convert = df_object.columns
    df_object = df_object[columns_to_convert]
    df_object[columns_to_convert] = df_object.apply(lambda x: x.str.strip())

    for c in columns_to_convert:
        print('%d %s' % (len(df[c].value_counts().to_dict()), c))


def get_useful_categorical(df, max_na=0.95, exclude_columns=[], min_in_class=100, min_classes=2, debug=True):
    categorical_columns = df.select_dtypes(include=['object']).columns.values
    categorical_columns = [x for x in categorical_columns if x not in exclude_columns]
    df_len = len(df)
    na_info = df[categorical_columns].isnull().sum() / df_len
    max_allowed_classes = round(df_len / min_in_class)

    if debug:
        print('There are %d categorical fields' % len(categorical_columns))
        print('Max allowed number of classes is %d' % max_allowed_classes)

    selected_columns = []
    for c in na_info.loc[na_info <= max_na].index:
        count_classes = len(df[c].value_counts())
        if min_classes <= count_classes < max_allowed_classes:
            selected_columns.append(c)
            if debug:
                print(c, count_classes)
        else:
            if debug:
                print('--', c, count_classes)

    if debug:
        print('Following %d fields will be skipped because of NA:\n' % len(na_info.loc[na_info > max_na]),
              na_info.loc[na_info > max_na])

    return selected_columns
