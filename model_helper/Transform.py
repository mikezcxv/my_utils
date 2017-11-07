from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Transform:
    map_le = {}

    def do(self, _x_train, _y_train):
        self.p1.fit(_x_train, _y_train)
        self.p2.fit(_x_train, _y_train)
        self.p3.fit(_x_train, _y_train)

    # Deal with categorical values
    def cast_multi_train(self, df, na_val='-999'):
        df_numeric = df.select_dtypes(exclude=['object'])
        df_obj = df.select_dtypes(include=['object']).copy()
        df_obj.fillna(na_val, inplace=True)

        for c in df_obj:
            self.map_le[c] = LabelEncoder()
            df_obj[c] = self.map_le[c].fit_transform(df_obj[c])

        return pd.concat([df_numeric, df_obj], axis=1)

    def cast_multi_test(self, df, na_val='-999'):
        df_numeric = df.select_dtypes(exclude=['object'])
        df_obj = df.select_dtypes(include=['object']).copy()
        df_obj.fillna(na_val, inplace=True)

        for c in df_obj:
            if c in df_obj.columns.values:
                df_obj[c] = self.map_le[c].transform(df_obj[c])

        return pd.concat([df_numeric, df_obj], axis=1)

    def cast_all_sets(self, df_train, df_test):
        return [self.cast_multi_train(df_train), self.cast_multi_test(df_test)]

