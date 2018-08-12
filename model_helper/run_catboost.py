from model_helper.visualize import *
from catboost import CatBoostClassifier
# import tqdm

def go(df, pk, target):
    '''
    Draft version
    :param df:
    :return:
    '''
    df_raw = df_all.copy()

    categorical_columns = get_useful_categorical(df_raw , max_na=0.9,
                                                 min_in_class=200, min_classes=3, debug=True)

    df_raw.sort_values(pk, ascending=True, inplace=True)
    df_raw.fillna(-999, inplace=True)

    df_raw.drop([date_column], axis=1, inplace=True)
    oos, l = .8, len(df_raw)

    data, y = df_raw.loc[:, df_raw.columns != target], df_raw.loc[:, target]
    train, labels = data.iloc[:round(l * oos), ], y.iloc[:round(l * oos), ]
    test, target_test = data.iloc[round(l * oos):, ], y.iloc[round(l * oos):, ]
    train, test = train.drop([pk], axis=1), test.drop([pk], axis=1)

    X_train = train.values
    X_test = test.values
    y_train = labels.values

    #         model_size_reg=None,
    #         border_count=None,
    #         fold_permutation_block_size=None,
    #         od_pval=None, od_wait=None, od_type=None,
    #         counter_calc_method=None,
    #         leaf_estimation_iterations, leaf_estimation_method,
    #         thread_count=None, logging_level=None,
    #         metric_period=None,
    #         ctr_leaf_count_limit=None,
    #         store_all_simple_ctr=None,
    #         max_ctr_complexity=None,
    #         has_time=None,
    #         class_weights=None,
    #         one_hot_max_size=None, random_strength=None,
    #         name=None,
    #         ignored_features,
    #         custom_loss, custom_metric,
    #         eval_metric=None,
    #         bagging_temperature=None,
    #         save_snapshot, snapshot_file,
    #         fold_len_multiplier=None,
    #         used_ram_limit=None,
    #         allow_writing_files=None,
    #         approx_on_full_history=None,
    #         simple_ctr=None,
    #         combinations_ctr=None,
    #         per_feature_ctr=None,
    #         ctr_description=None,
    #         task_type=None,
    #         device_config=None,


    count_runs = 2
    predsCB2 = list(np.zeros(len(target_test)))
    for i in tqdm(range(count_runs)):
        #     learning_rate=0.1, depth=4, iterations=220,

        #     iterations = 120 + (i * 10)
        #     learning_rate = 0.15 - (i / 200)
        #     depth = min(4, math.floor(i / 2) + 1)
        iterations = 350
        learning_rate = 0.1
        depth = 4

        model = CatBoostClassifier(verbose='Silent',
                                   iterations=iterations, learning_rate=learning_rate, depth=depth,
                                   loss_function='MultiClass', eval_metric='AUC', l2_leaf_reg=3, random_seed=i + 1)
        model.fit(X_train, y_train, plot=(i == 0), verbose=False,
                  cat_features=[train.columns.get_loc(c) for c in
                                categorical_columns])

        # cat_columns + cat_columns_clarity

        # !TODO save model to file
        #
        pr = model.predict_proba(X_test)[:, 1]
        print(roc_auc_score(target_test, pr))
        predsCB2 += pr

    predsCB2 /= count_runs
    print(roc_auc_score(target_test, predsCB2))
