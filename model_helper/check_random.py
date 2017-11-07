from model_helper.stat import *


def test_random_var(df, is_prepare_stat=False):
    # TODO remove
    _y = df.ix[:, 'SecondPaymentDelinquencyClass']
    r_iv = []
    r_auc = []
    # Try several random categories
    for i in range(1, 500):
        print('.', end='')
        c_name = 'random' + str(i)
        # df[c_name] = np.random.choice(2, df.shape[0])
        df[c_name] = np.random.choice(range(100, np.random.randint(10000, 1000000)), df.shape[0]) / 100

        # remove part of data
        replace_from = np.random.randint(1, int(len(df) / 5))
        df.loc[df.index.values[replace_from:10000], c_name] = np.NAN

        # df[c_name] = np.random.choice(5, df.shape[0])
        if is_prepare_stat:
            r_iv.append(iv2(df, c_name, _y))
            r_auc.append(roc_auc_score(_y, df[c_name]))
            print("IV of random var %.4f: " % iv2(df, c_name, _y))
            print("Auc: %.4f" % roc_auc_score(_y, df[c_name]))

    if is_prepare_stat:
        print(stats.describe(np.array(r_iv)))
        print(stats.describe(np.array(r_auc)))
