# 完成插入并更新操作
import pymysql
import numpy as np
import pandas as pd

class MySQL:
    def __init__(self, host, port, user, pwd, db):
        self.host = host
        self.port = int(port)
        self.user = user
        self.pwd = pwd
        self.db = db
        if not self.db:
            raise NameError("没有设置数据库信息")
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                passwd=self.pwd,
                db=self.db
            )
        except Exception as e:
            print(e)
            raise AttributeError("连接数据库失败")

        self.cur = self.conn.cursor()
        if not self.cur:
            raise NameError("获取游标失败")

    def execQuery(self, query):
        # 查询语句请用 pandas.read_sql()
        self.cur.execute(query)
        resList = self.cur.fetchall()
        # 查询完毕必须关闭连接
        return resList

    def execNonQuery(self, sql):
        self.cur.execute(sql)
        self.conn.commit()

    def ex(self):
        self.conn.close()


def save_to_sql(conn, data, table, ifclose=True, chunksize=8000):
    '''
    Params:
        conn:
            - instance of MySQL class or MSSQL class
        data:
            - DataFrame
            --Note: data的列名必须与表列名一致(不区分大小写)
        table:
            - str, 要插入的表名称
    '''
    conn.execNonQuery('USE %s' % conn.db)

    # 获取字段名称
    data = data.reset_index(drop=True)
    data_col_name = [col.lower() for col in data.columns.values]
    rows = len(data)
    tab_col = pd.read_sql('DESCRIBE %s' % table, conn.conn, index_col='Field')
    tab_col.index = tab_col.index.str.lower()
    tab_col_type = tab_col.Type.to_dict()
    chunk_num = int(np.ceil(rows / chunksize))
    for cn in range(chunk_num):
        string = []
        for dcn in data_col_name:
            tp = tab_col_type[dcn]
            if tp.lower()[:8] == 'datetime':
                temp_str = r"str_to_date('%s','%%Y-%%m-%%d %%H:%%i:%%s')"
            elif tp.lower()[:4] == 'date':
                temp_str = r"str_to_date('%s','%%Y-%%m-%%d')"
            else:
                temp_str = r"'%s'"
            if dcn == data_col_name[0]:
                temp_str = '(' + temp_str
            elif dcn == data_col_name[-1]:
                temp_str = temp_str + ')'
            string.append(temp_str)

        chunk_data = data.iloc[cn * chunksize:(cn + 1) * chunksize] if cn < chunk_num - 1 else data.iloc[
                                                                                               cn * chunksize:]

        temp_sql = ','.join(string * len(chunk_data)) % tuple(chunk_data.values.ravel())

        temp_sql = temp_sql.replace('nan', 'null')
        temp_sql = temp_sql.replace('\'null\'', 'null')
        insert_col = ','.join(['`%s`' % col for col in data_col_name])
        update_col = ','.join(['`%s`=Values(`%s`)' % (col, col) for col in data_col_name])
        sql = 'INSERT INTO %s.%s (%s) VALUES ' % (conn.db, table, insert_col) + \
              temp_sql + ' ON DUPLICATE KEY UPDATE ' + update_col + ';'

        sql = sql.encode('utf-8')
        conn.execNonQuery(sql)
    if ifclose:
        conn.ex()

