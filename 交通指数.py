def trafficDBdata():
    # 计算交通指数并入库
    stdsql = insert_update.MySQL('10.181.31.154', 3310, 'cdev', 'Dev154@industry', 'industry')
    sql = 'SELECT * FROM traffic_real_index'
    dataframes = pd.read_sql(sql, con=stdsql.conn)
    db = dataframes[['id', 'time', 'road_flood', 'road_wind', 'road_ice']]
    timeList = sorted(pd.to_datetime(db.time.unique()).strftime('%Y-%m-%d %H:%M:%S'))
    for index in timeList:
        flood = db.loc[index].road_flood.values[:, np.newaxis]   #
        wind = db.loc[index].road_wind.values[:, np.newaxis]
        ice = db.loc[index].road_ice.values[:, np.newaxis]
        traffic = pd.DataFrame(np.max(np.concatenate([flood,wind,ice], axis=1), axis=0), columns=['road_safety'])

        id  =  db.loc[index].id.values[:, np.newaxis]     # id
        times = db.loc[index].time.values[:, np.newaxis]  # 时间
        res = pd.DataFrame(np.concatenate([id, times, traffic], axis=1), columns=['id', 'time', 'road_safety'])
        insert_update.save_to_sql(stdsql, res, 'traffic_real_index')
    return None