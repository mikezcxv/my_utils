import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd

# n = 10010
# c = np.linspace(21, n + 20, n)
# c[105] = np.NAN
#
# df = pd.DataFrame({'a': np.linspace(1, n, n), 'b': np.linspace(11, n + 10, n), 'c': c})
#
# df.to_csv('test.csv', index=False)

# pd.read_csv('test.csv')

# df_path = 'test.csv'
# num_rows = 5000


def show_memory_usage(df):
    m = df.memory_usage(True, deep=True).sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", m, " MB")
    return m


class GetDfTypes:
    def __init__(self, df_path, read_rows=10 ** 3, convert_float=True, sep=',', na_values=[-1, '-1']):
        self.df_path = df_path
        int_columns, float_columns = self.group_df_columns_info(read_rows, sep, na_values)
        self.types = self.define_types(int_columns, float_columns if convert_float else {})

        # For debug
        # print(int_columns, float_columns)
        # for c in df_full.columns.values:
        #     print(c, np.min(df_full[c]), np.max(df_full[c]))

    def get_list(self):
        return self.types

    # Inspired by
    # https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/notebook
    @staticmethod
    def define_types(int_list, float_list):
        types = {}
        for col, v in int_list.items():
            mx, mn = v['max'], v['min']

            # Make Integer/unsigned Integer data types
            if mn >= 0:
                if mx < np.iinfo(np.uint8).max:
                    types[col] = 'uint8'
                elif mx < np.iinfo(np.uint16).max:
                    types[col] = 'uint16'
                elif mx < np.iinfo(np.uint32).max:
                    types[col] = 'uint32'
                else:
                    types[col] = 'uint64'
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    types[col] = 'int8'
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    types[col] = 'int16'
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    types[col] = 'int32'
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    types[col] = 'int64'

        for col, v in float_list.items():
            types[col] = 'float32'

        return types

    @staticmethod
    def update_min(_list, _chunk, _c):
        _list[_c]['min'] = min(_list[_c]['min'], np.min(_chunk[_c]))

    @staticmethod
    def update_max(_list, _chunk, _c):
        _list[_c]['max'] = max(_list[_c]['max'], np.max(_chunk[_c]))

    def group_df_columns_info(self, read_rows, sep, na_values):
        int_columns, float_columns = {}, {}

        chunks = pd.read_csv(self.df_path, chunksize=read_rows, sep=sep, na_values=na_values)
        for i, chunk in enumerate(chunks):

            float_cols = list(chunk.select_dtypes(include=['float']).columns)
            int_cols = list(chunk.select_dtypes(include=['int']).columns)

            for c in int_cols:
                if c not in int_columns:
                    # The first occurrence in int list. If not in floats
                    if c not in float_columns:
                        int_columns[c] = {'min': np.min(chunk[c]), 'max': np.max(chunk[c])}
                    else:
                        self.update_min(float_columns, chunk, c), self.update_max(float_columns, chunk, c)
                else:
                    self.update_min(int_columns, chunk, c), self.update_max(int_columns, chunk, c)

            for c in float_cols:
                if c not in float_columns:
                    if c in int_columns:
                        del int_columns[c]

                    float_columns[c] = {'min': np.min(chunk[c]), 'max': np.max(chunk[c])}
                else:
                    self.update_min(float_columns, chunk, c), self.update_max(float_columns, chunk, c)

        return int_columns, float_columns


# df_full = pd.read_csv(df_path)
# show_memory_usage(df_full)
#
# types_list = GetDfTypes(df_path).get_list()
# print(types_list, df_full.head(), df_full.dtypes)
#
# df_full2 = pd.read_csv(df_path, dtype=types_list)
# show_memory_usage(df_full2)
#
# print(df_full2.head(), df_full2.dtypes)

# types[col] = 'uint8'
# elif mx < 65535:
# types[col] = 'uint16'
# elif mx < 4294967295:
# types[col] = 'uint32'
# else:
# types[col] = 'uint64'

# , df_full.head()
# print(int_columns)
# print(float_columns)

# chunks = pd.read_csv(df_path, chunksize=num_rows)
# for i, chunk in enumerate(chunks):
    # float_cols = list(chunk.select_dtypes(include=['float']).columns)
    # int_cols = list(chunk.select_dtypes(include=['int']).columns)

    # print(chunk['a'])

    # chunk['a'] = chunk['a'].astype('int8')

    # print(chunk['a'])
    # print(chunk.dtypes)
    # chunk.ix[10, 'a'] = 1.0
    # print(chunk.dtypes)
    # break


# print(df.mem)

# print(df.ix[0:1, ].dtypes)

# np.random.seed(42)
# l = np.random.normal(0, 10, 1000)
#
# print(np.std(l), np.min(l), np.max(l))
#
# l = np.append(l, 410)
#
# # plt.hist(l)
# # plt.show()
#
# print(np.std(l), np.min(l), np.max(l))
