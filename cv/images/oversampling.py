import pandas as pd
import numpy as np


PREV_CHUNK_INDEX = 0


def glue_df_gen2(df_source, df_prev_chunks, replace_pairs, oversample_new_dataset, is_add_prev_data,
                 is_remove_duplicates_from_train, is_stratified=False, debug_percentage=None,
                 oversample_rectangular=True, shrink_validation_bound=False, fr_prev=.1, df_sub=None):
    cols = ['id_code', target, 'size_ratio']
    fr = debug_percentage if debug_percentage else 1

    df = df_source.sample(frac=fr, random_state=42).reset_index()[cols]

    df_prev_adj = []
    for i in range(PREV_CHUNK_INDEX):
        df_prev_adj.append([])  # TODO refactor. Adding tmp dummy []
    df_prev_adj.append(
        df_prev_chunks[PREV_CHUNK_INDEX].sample(frac=min(fr_prev, fr), random_state=42).reset_index()[cols])

    if not is_stratified:
        indices = list(range(len(df)))
        kf = KFold(n_splits=n_folds - 1 if shrink_validation_bound else n_folds, random_state=42, shuffle=True)
    else:
        raise NotImplemented()

    train_idx, valid_idx = [], []
    for t, v in kf.split(indices):
        trn_idx, val_idx = t, v

        train, valid = df.loc[t], df.loc[v]
        df['oversampled'] = 0
        df['train'] = 1
        df.loc[v, 'train'] = 0
        if oversample_new_dataset == 'more':
            oversample_new_dataset = [1, 2, 1, 3, 3]
        elif oversample_new_dataset == 'less':
            oversample_new_dataset = [0, 1, 0, 1, 1]
        elif type(oversample_new_dataset) == list or oversample_new_dataset is None:
            pass
        else:
            raise NotImplementedError('NotImplementedError has 3 possible arguments: None, more, less')

        if type(oversample_new_dataset) == list:
            oversample = [df]
            for i, n_times in enumerate(oversample_new_dataset):
                tmp_df = df[(df['train'] == 1) & (df[target] == i)]
                tmp_df['oversampled'] = 1
                for j in range(n_times):
                    oversample.append(tmp_df)

            df = pd.concat(oversample, axis=0)
            del oversample, tmp_df
            print(f'Oversampled train data size {df.shape[0]}')

        if is_add_prev_data:
            print('Adding data from previous competitions')
            print(f'Train shape {df.shape[0]}')
            df_prev_adj[PREV_CHUNK_INDEX]['train'] = 1
            df = pd.concat([df, df_prev_adj[PREV_CHUNK_INDEX]], axis=0)
            print(dict(df[target].value_counts()))
        else:
            print('!!! previous competitions data are not added !!!')

        if oversample_rectangular:
            df_rectangular = df[(df['train'] == 1) & (df['size_ratio'] >= 1.3)]
            df = pd.concat([df, df_new_train, df_minority, df_rectangular], axis=0)
            print(f'Oversampled by rectangular {df.shape[0]}')

        if not df_sub is None:
            df = pd.concat([df, df_sub], axis=0)
            print(f'Oversampled by pseudolabels {df.shape[0]}')

        if oversample_new_dataset or is_add_prev_data:
            df = df.reset_index()[['id_code', target, 'size_ratio', 'train', 'oversampled']]
            trn_idx = df[df['train'] == 1].index
            val_idx = df[df['train'] == 0].index
            train, valid = df.loc[trn_idx], df.loc[val_idx]

        if shrink_validation_bound:
            print('In validation before shrink: ', len(df[df['train'] == 0]))
            df['train'] = np.where((df['train'] == 0) & (df['size_ratio'] < 1.2), 1, df['train'])
            print('In validation after shrink: ', len(df[df['train'] == 0]))

            trn_idx = df[df['train'] == 1].index
            val_idx = df[df['train'] == 0].index
            train, valid = df.loc[trn_idx], df.loc[val_idx]

        if is_remove_duplicates_from_train:
            tmp_v = valid['id_code'].apply(lambda x: x.split('/')[-1])
            duplicates_in_validation = tmp_v[tmp_v.isin(replace_pairs.keys())]
            duplicates_to_remove_from_train = [replace_pairs[x] for x in duplicates_in_validation]

            tmp = train.copy()
            tmp['id_code'] = tmp['id_code'].apply(lambda x: x.split('/')[-1])

            print('train shape', train.shape)
            train = train[~train.index.isin(tmp[tmp['id_code'].isin(duplicates_to_remove_from_train)].index)]
            trn_idx = train.index
            del tmp_v, tmp
            print('df after duplicates removal: ', train.shape)

        train_idx.append(trn_idx)
        valid_idx.append(val_idx)

        break

    return df, train_idx, valid_idx
