import warnings
import wandb
import numpy as np
import FinanceDataReader as fdr
import matplotlib
from spillover import SpillOver

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    matplotlib.use('Agg')

    start = '2016-01-01'
    sp_total = fdr.DataReader('US500', start=start)
    bt_total = fdr.DataReader('BTC/USD', start=start)
    final_date = sp_total.index[-1]
    tdelta = 365
    fevd_horizon = 10
    fevd_target = [3, 6, 9]

    wandb.init(
        entity='db22',
        project='SpillOver',
        name=f'vol {tdelta}',
        config={
            'start': start,
            'period': tdelta,
            'fevd_horizon': fevd_horizon
        })

    modes = ['vol', 'var05', 'var01']
    names = ['S&P500', 'BTC_USD']
    mu_params = {
        'p': [0, 1, 2],
        'd': [0, 1],
        'q': [0, 1, 2],
        'P': [0],
        'D': [0],
        'Q': [0],
        'S': [0]
    }
    resid_params = {
        'p': [1, 2],
        'o': [0, 1],
        'q': [1, 2],
        'dist': ['normal', 'studentsT', 'T', 'skewT']
    }
    model = SpillOver(sp_total, bt_total, start, tdelta)
    step = 0
    while True:
        eda_result = model.modeling(names, mu_params, resid_params)
        tmp_ls = []
        for mode in modes:
            try:
                # AIC : low model-size penalty
                # BIC : increases the penalty as the sample size increases
                net_df, pvalue1, pvalue2 = model.run(mode, names, VAR_maxlags=None, VAR_ic='bic',
                                                     fevd_target=fevd_target,
                                                     fevd_horizon=fevd_horizon)
                # For wandb
                log_dict = {
                    f'{mode} GC BTC/USD -> S&P500': pvalue1,
                    f'{mode} GC S&P500 -> BTC/USD': pvalue2,
                    f'{mode} VAR lags': model.VAR_lags[-1][-1]
                }
                log_dict.update({f'{mode} S&P500 NET {fevd_target[i]}th horizon': net_df.iloc[i, 0] for i in
                                 range(len(fevd_target))})
                log_dict.update({f'{mode} BTC/USD NET {fevd_target[i]}th horizon': net_df.iloc[i, 1] for i in
                                 range(len(fevd_target))})
                wandb.log(log_dict, step=step)

                tmp_ls.append(net_df.values)
            except Exception as e:
                print("*********Failures!*********")
                print(model.period, mode)
                print(e)
                print("***************************")

        model.net_spillover_ls.append(tmp_ls)
        print('=' * 20)
        print(model.period, 'Done!')
        print('=' * 20)
        if model.end == final_date:
            break
        model.update_date()
        step += 1

    net_spillover_ls = np.array(model.net_spillover_ls)
    np.save('result/net_spillover.npy', net_spillover_ls)

