
import  datetime
from    sklearn.preprocessing       import RobustScaler

### 時間の取得
def get_now():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))) # 日本時刻
    return now.strftime('%Y%m%d%H%M%S')

### 特徴量の作成
def make_feature(df_train, df_test, drop_ls):
    
    df_train    = df_train.drop(drop_ls, axis=1)
    df_test     = df_test.drop(drop_ls, axis=1)

    # df_train, df_test = scaling(df_train,df_test)
    df_train    = add_sum_feature(df_train)
    df_test     = add_sum_feature(df_test)
    
    return df_train, df_test

### 関連特徴量の和を新たに特徴量に加える
def add_sum_feature(df):
    pfx_ls = ['co','o3','so2',\
            'temperature','humidity',\
            'pressure','ws','dew']
    sfx_ls = ['_cnt', '_min', '_mid', '_max', '_var']

    for pfx in pfx_ls:
        ### num番目に0で初期化した列を追加
        new_col = pfx+'_sum'

        num = df.columns.get_loc(pfx+sfx_ls[-1]) + 1
        df.insert(num, new_col, 0)

        for sfx in sfx_ls:
            tmp_col     = pfx+sfx
            df[new_col] = df[new_col] + df[tmp_col]
            num += 1

    return df

### ロバストスケーリングを行う
def scaling(df1, df2):
    scaler  = RobustScaler(quantile_range=(25.0, 75.0))
    features= df2.columns.to_list()

    scaler.fit(df1[features])
    df1[features]  = scaler.transform(df1[features])
    df2[features]  = scaler.transform(df2[features])   
         
    return df1,df2

