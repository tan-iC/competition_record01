
import  os
import  torch
import  matplotlib
matplotlib.use('Agg')
import  pandas                      as pd
import  matplotlib.pyplot           as plt
from    utilcode                    import *
from    torch._C                    import device
from    operator                    import itemgetter
from    sklearn.model_selection     import KFold
from    pytorch_tabnet.tab_model    import TabNetRegressor
from    pytorch_tabnet.pretraining  import TabNetPretrainer


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda')
print(device)

###
### データの読み込み
###
train   = pd.read_csv('../data/train.csv')
test    = pd.read_csv('../data/test.csv')

### 変数リスト
vars_ls = train.columns.to_list()
print(vars_ls)
'''
'id', 'year', 'month', 'day', 'Country', 'City', 'lat', 'lon', 
'co_cnt', 'co_min', 'co_mid', 'co_max', 'co_var', 
'o3_cnt', 'o3_min', 'o3_mid', 'o3_max', 'o3_var', 
'so2_cnt', 'so2_min', 'so2_mid', 'so2_max', 'so2_var', 
'no2_cnt', 'no2_min', 'no2_mid', 'no2_max', 'no2_var', 
'temperature_cnt', 'temperature_min', 'temperature_mid', 
'temperature_max', 'temperature_var', 'humidity_cnt', 
'humidity_min', 'humidity_mid', 'humidity_max', 'humidity_var', 
'pressure_cnt', 'pressure_min', 'pressure_mid', 'pressure_max', 
'pressure_var', 'ws_cnt', 'ws_min', 'ws_mid', 'ws_max', 'ws_var', 
'dew_cnt', 'dew_min', 'dew_mid', 'dew_max', 'dew_var', 'pm25_mid'
'''

###
### 目的変数名
###
target  = vars_ls[-1]

###
### 特徴量除外する変数名リスト
###
drop_ls = list(itemgetter(0,4,5)(vars_ls))

###
### ハイパーパラメータ
###
SEED    = 53
N_STEPS = 10
rand_st = 0
n_d     = 64
n_a     = 32
n_split = 5
pol_ls  = ['drop', 'scale', 'valid', 'sum', 'kfold']
policy  = f'{pol_ls[0]}_{pol_ls[4]}_{pol_ls[3]}_{n_d}x{n_a}'

###
### 特徴量作成
###
# print(f'train[0:3]:\n{train[0:3]}')
print('make feature...')

train, test_X   = make_feature(train, test.copy(), drop_ls)
# print('')
# print(f'train[0:3]:\n{train[0:3]}')

train_X = train.iloc[:, :-1].values
train_Y = train.iloc[:, [-1]].values
test_X  = test_X.values

# print(f'train.shape:{train_X.shape}, {train_Y.shape}')
# print(f'test.shape:{test_X.shape}')
print('preprocessing done!')


###
### tabnetを用いた学習
###【参考】（https://qiita.com/maskot1977/items/5de6605806f8918d2283）
###

###
### パラメータ設定
###
tabnet_params = dict(n_d=n_d, n_a=n_a, n_steps=N_STEPS, gamma=1.3,
                     n_independent=2, n_shared=2,
                     device_name='cuda',
                     seed=SEED, lambda_sparse=1e-3, 
                     optimizer_fn=torch.optim.Adam, 
                     optimizer_params=dict(lr=2e-2),
                     mask_type="entmax",
                     scheduler_params=dict(mode="min",
                                           patience=5,
                                           min_lr=1e-5,
                                           factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10
                    )

###
### 事前学習開始
###
print('start pretraining...')
pretrainer = TabNetPretrainer(**tabnet_params)
pretrainer.fit(
    X_train=train_X,
    eval_set=[train_X],
    max_epochs=5000,
    patience=100,
    batch_size=1024*8
    )


now = get_now()
print(f'{now} pretraining done!')
saving_path_name    = f"../result/pretrainer_{policy}_{now}"
saved_filepath      = pretrainer.save_model(saving_path_name)


###
### 学習経過の出力
###
for param in ['loss', 'lr']:
    plt.figure()
    plt.plot(pretrainer.history[param])
    plt.xlabel('epoch')
    plt.ylabel(param)
    plt.grid()
    plt.savefig(f'../result/{now}-{param}-{policy}-pre.png')


###
### (教師あり)学習開始
###
print('start learning...')
model = TabNetRegressor(**tabnet_params)

###
### 【参考】
###     (https://qiita.com/t-smz/items/6e5d6c10aba7a8e3f991)
###     (https://www.kaggle.com/code/tamreff3290/tabnet-titanic/notebook)
###
###
kf = KFold(n_splits=n_split, shuffle=False)
n_fold = 0
scores = []
for train_index, valid_index in kf.split(train_X, train_Y):
    x_tr    = train_X[train_index]
    x_val   = train_X[valid_index]
    y_tr    = train_Y[train_index]
    y_val   = train_Y[valid_index]
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        patience=100,
        max_epochs=500,
        batch_size=1024*8,
        eval_metric={'rmse'}, # 'rmsle', 'mse', 'mae'
        from_unsupervised=pretrainer
    )

    if n_fold == n_split - 1:
        ###
        ### 学習結果の出力
        ###
        for name, X, y in [["training", x_tr, y_tr], ["validation", x_val, y_val]]:
            plt.figure()
            plt.scatter(y, model.predict(X), alpha=0.5, label=name)
            plt.plot()
            plt.grid()
            plt.legend()
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.savefig(f'../result/{now}-{name}-{policy}.png')
    n_fold += 1


###
### スコア
###
print('learning done!')
print(f"BEST VALID SCORE: {model.best_cost}")


###
### 学習経過の出力
###
for param in ['loss', 'lr', 'val_0_rmse']:
    plt.figure()
    plt.plot(model.history[param], label=param)
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.savefig(f'../result/{now}-{param}-{policy}.png')


###
### モデルの保存
###
saving_path_name    = f"../result/tabnet_{policy}_{now}"
saved_filepath      = model.save_model(saving_path_name)


###
### 予測
###
pred = model.predict(test_X)
pred = pd.Series(pred.T[0])
sbmt = pd.DataFrame({'id': test.iloc[:,0], 'pred':pred})
sbmt.to_csv(f'../result/sbmt-pre-{now}.csv', index=False, header=False)
