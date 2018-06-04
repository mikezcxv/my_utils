import psycopg2
import re
from psycopg2.extensions import AsIs
from datetime import datetime

# Usage Example
# db_write_helper = DbWriter(conn_credentials, WRITE_TABLE, pk)
#
# for line in lines:
#     data_map = {
#         'item_id': pk_val,
#         'raw_data': line,
#         ...
#     }
#     db_write_helper.insert_or_update(pk_val, data_map)
#
# db_write_helper.finalize()


class DbWriter:

    def __init__(self, credentials, table_name, pk_name, commit_every_n=200,
                 default_create_column=None,
                 default_update_column=None):
        self.conn = None
        self.cur_insert = None
        self.connect(credentials)
        self.recorded = 0
        self.commit_every_n = max(1, commit_every_n)
        self.need_commit = True
        self.table_name = table_name
        self.pk_name = pk_name
        self.default_create_column = default_create_column
        self.default_update_column = default_update_column

    def connect(self, credentials):
        try:
            self.conn = psycopg2.connect(credentials)
            self.cur_insert = self.conn.cursor()
        except ConnectionError:
            print("Unable to connect to the database")

    @staticmethod
    def normalize_column_name(raw_column_name):
        _name = re.sub(r"[.\[\]]", "_", raw_column_name)
        return re.sub(r"_+", "_", _name)

    def row_exists(self, val, column_name):
        _cur = self.conn.cursor()
        _cur.execute("SELECT 1 FROM " + self.table_name + " WHERE " + column_name + " = %s", (val,))
        return _cur.fetchone() is not None

    def perform_commit(self):
        if self.recorded % self.commit_every_n == 0:
            self.conn.commit()
            self.need_commit = False
        else:
            self.need_commit = True

    def insert_or_update(self, pk_val, data):
        if self.row_exists(pk_val, self.pk_name):

            if self.default_update_column:
                data[self.default_update_column] = datetime.now()

            d = data.copy()
            del d[self.pk_name]
            prepared_set, placeholders = [], []
            for k, v in d.items():
                prepared_set.append(k + ' = %s')
                placeholders.append(v)

            update_stmt = "UPDATE " + self.table_name + " SET " + (', '.join(prepared_set)) \
                          + " WHERE " + self.pk_name + " = '%s'" % str(pk_val)
            self.cur_insert.execute(update_stmt, placeholders)
        else:
            if self.default_create_column:
                data[self.default_create_column] = datetime.now()

            columns = data.keys()
            values = [data[column] for column in columns]

            insert_statement = 'INSERT INTO ' + self.table_name + ' (%s) values %s'
            self.cur_insert.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))

        self.perform_commit()

    def finalize(self):
        self.conn.commit()
        self.cur_insert.close()
        self.conn.close()

    def __del__(self):
        self.finalize()


class Queries:
    def __init__(self, credentials, table_name):
        self.conn = self.get_connect(credentials)
        self.table_name = table_name

    @staticmethod
    def get_connect(credentials):
        try:
            return psycopg2.connect(credentials)
        except ConnectionError:
            print("Unable to connect to the database")

    def fetch_row(self, val, column_name, columns):
        columns = ','.join(columns)
        _cur = self.conn.cursor()
        _cur.execute("SELECT " + columns + " FROM %s WHERE %s = %s"
                     % (self.table_name, column_name, val))
        return _cur.fetchone()

    def execute(self, query):
        _cur = self.conn.cursor()
        _cur.execute(query)
        return self.conn.commit()
