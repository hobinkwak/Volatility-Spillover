import os
import random
import datetime as dt
import pickle as pkl
from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import statsmodels
import statsmodels.api as sm



class SpillOver:
    def __init__(self, data1, data2, start, tdelta):
        self.total1 = data1
        self.total2 = data2
        self.start = pd.to_datetime(start)
        self.tdelta = tdelta
        self.end = self.start + dt.timedelta(self.tdelta)
        self.period = f"{str(self.start)[:10]} ~ {str(self.end)[:10]}"
        os.makedirs(f'result/{self.period}', exist_ok=True)
        self.net_spillover_ls = []
        self.VAR_lags = []
        np.random.seed(97)
        random.seed(97)

    def update_date(self):
        self.start = self.start + dt.timedelta(1)
        self.end = self.end + dt.timedelta(1)
        self.period = f"{str(self.start)[:10]} ~ {str(self.end)[:10]}"
        os.makedirs(f'result/{self.period}', exist_ok=True)

    def set_subdata(self):
        data1 = self.total1.loc[self.start:self.end, :]
        data2 = self.total2.loc[self.start:self.end, :]
        return data1, data2

    def compute_return(self, data1, data2):
        # scale by 100 for convergence of optimizer
        rtn1 = np.log(data1.Close / data1.Close.shift(1)).dropna() * 100
        rtn2 = np.log(data2.Close / data2.Close.shift(1)).dropna() * 100
        return rtn1, rtn2

    def compute_vol(self, rtn1, rtn2):
        daily_sp_std = rtn1.std()
        daily_bt_std = rtn2.std()
        monthly_sp_std = rtn1.std() * np.sqrt(21)
        monthly_bt_std = rtn2.std() * np.sqrt(30)
        yearly_sp_std = rtn1.std() * np.sqrt(252)
        yearly_bt_std = rtn2.std() * np.sqrt(365)
        f = open(f'result/{self.period}/volatility.txt', 'w')
        f.write(f"일일 S&P500 변동성: {daily_sp_std} \n")
        f.write(f"일일 BTC/USD 변동성: {daily_bt_std} \n")
        f.write(f"월간 S&P500 변동성: {monthly_sp_std} \n")
        f.write(f"월간 BTC/USD 변동성: {monthly_bt_std} \n")
        f.write(f"연간 S&P500 변동성: {yearly_sp_std} \n")
        f.write(f"연간 BTC/USD 변동성: {yearly_bt_std} \n")
        f.close()

    def describe(self, rtn, name):
        desc = rtn.describe().T
        desc['skew'] = rtn.skew()
        desc['kurt'] = rtn.kurtosis()
        desc.to_csv(f'result/{self.period}/{name}_desc.csv')
        return desc

    def normality_test(self, rtn, name):
        """
        null hypotheis : sample is drawn from normal distribution
        """
        normality = {}
        normality['shapiro statistic'] = stats.shapiro(rtn).statistic
        normality['shapiro p-value'] = stats.shapiro(rtn).pvalue
        with open(f'result/{self.period}/{name}_normality_test.pkl', 'wb') as f:
            pkl.dump(normality, f)
        return normality

    def autocorr_test(self, rtn, name):
        """
        null hypothesis : sample is white noise
        """
        ljung = sm.stats.acorr_ljungbox(rtn, return_df=True)
        ljung.to_csv(f'result/{self.period}/{name}_ljung_box_test.csv')
        return ljung

    def stationarity_test(self, rtn, name):
        """
        adf null hypothesis : not stationary
        kpss null hypothesis : stationary
        """
        stationarity = {}
        stationarity['adf statistic'] = sm.tsa.stattools.adfuller(rtn)[0]
        stationarity['adf p-value'] = sm.tsa.stattools.adfuller(rtn)[1]
        stationarity['kpss statistic'] = sm.tsa.stattools.kpss(rtn)[0]
        stationarity['kpss p-value'] = sm.tsa.stattools.kpss(rtn)[1]
        with open(f'result/{self.period}/{name}_stationarity_test.pkl', 'wb') as f:
            pkl.dump(stationarity, f)
        return stationarity

    def histogram(self, rtn, name):
        sns.distplot(rtn, bins=30)
        plt.title(name + self.period)
        plt.savefig(f'result/{self.period}/{name}_histogram.png')
        plt.close()

    def plot_autocorr(self, rtn, name):
        statsmodels.graphics.tsaplots.plot_acf(rtn)
        plt.savefig(f'result/{self.period}/{name}_acf.png')
        plt.close()
        statsmodels.graphics.tsaplots.plot_pacf(rtn)
        plt.savefig(f'result/{self.period}/{name}_pacf.png')
        plt.close()

    def eda(self, rtn1, rtn2, names):
        self.compute_vol(rtn1, rtn2)
        self.describe(rtn1, names[0])
        self.describe(rtn2, names[1])
        rtn1_normal = self.normality_test(rtn1, names[0])
        rtn2_normal = self.normality_test(rtn2, names[1])
        rtn1_autocorr = self.autocorr_test(rtn1, names[0])
        rtn2_autocorr = self.autocorr_test(rtn2, names[1])
        rtn1_stationary = self.stationarity_test(rtn1, names[0])
        rtn2_stationary = self.stationarity_test(rtn2, names[1])
        self.histogram(rtn1, names[0])
        self.histogram(rtn2, names[1])
        self.plot_autocorr(rtn1, names[0])
        self.plot_autocorr(rtn2, names[1])
        return [rtn1_normal, rtn2_normal, rtn1_autocorr, rtn2_autocorr, rtn1_stationary, rtn2_stationary]

    def set_mu_parameter(self, p, d, q, P, D, Q, S):
        trend_pdq = list(product(p, d, q))
        seasonal_pdq = [(param[0], param[1], param[2], param[3]) for param in list(product(P, D, Q, S))]
        return trend_pdq, seasonal_pdq

    def search_mu_parameter(self, rtn, trend_pdq, seasonal_pdq):
        bic_ls = []
        param_ls = []
        for trend_param in trend_pdq:
            for seasonal_params in seasonal_pdq:
                for trend in ['c', 'nc']:
                    try:
                        model = sm.tsa.SARIMAX(rtn, trend=trend,
                                               order=trend_param, seasonal_order=seasonal_params)
                        res = model.fit(method='powell', maxiter=1000, disp=False)
                        bic_ls.append(res.bic)
                        param_ls.append([trend_param, seasonal_params, trend])
                    except:
                        continue
        bic_ls = np.array(bic_ls)
        best_param = param_ls[np.argmin(bic_ls)]
        return best_param

    def mu_modeling(self, rtn, name, p, d, q, P, D, Q, S, summary=True):
        trend_pdq, seasonal_pdq = self.set_mu_parameter(p, d, q, P, D, Q, S)
        best_param = self.search_mu_parameter(rtn, trend_pdq, seasonal_pdq)
        with open(f'result/{self.period}/mu_best_param.pkl', 'wb') as f:
            pkl.dump(best_param, f)
        try:
            mu_model = sm.tsa.SARIMAX(rtn, trend=best_param[2],
                                      order=best_param[0], seasonal_order=best_param[1])
            mu_res = mu_model.fit(method='powell', maxiter=1000, disp=False)
        except:
            mu_model = sm.tsa.SARIMAX(rtn, trend='nc',
                                      order=best_param[0], seasonal_order=best_param[1])
            mu_res = mu_model.fit(method='powell', maxiter=1000, disp=False)
        if summary:
            print(mu_res.summary())
        mu_fittedvalues = mu_res.fittedvalues
        mu_resid = mu_res.resid
        resid_normal = self.normality_test(mu_resid, name + 'resid')
        resid_autocorr = self.autocorr_test(mu_resid, name + 'resid')
        resid_stationary = self.stationarity_test(mu_resid, name + 'resid')
        self.histogram(mu_resid, name + 'resid')
        self.plot_autocorr(mu_resid, name + 'resid')
        return mu_fittedvalues, mu_resid, [resid_normal, resid_autocorr, resid_stationary]

    def search_resid_parameter(self, resid, p, o, q, dist):
        params_ls = []
        garch_bic = []
        for params in product(p, o, q, dist):
            if (params[0] == 0) & (params[1] == 0):
                continue
            else:
                try:
                    vol_model = arch_model(resid, vol='GARCH', mean='zero', p=params[0]
                                           , o=params[1], q=params[2], dist=params[3])
                    vol_res = vol_model.fit(disp='off', options={'maxiter':1000})
                    garch_bic.append(vol_res.bic)
                    params_ls.append(params)
                except:
                    continue

        best_params = params_ls[np.array(garch_bic).argmin()]
        return best_params

    def resid_modeling(self, resid, p, o, q, dist, name, summary=True):
        best_params = self.search_resid_parameter(resid, p, o, q, dist)
        with open(f'result/{self.period}/resid_best_param.pkl', 'wb') as f:
            pkl.dump(best_params, f)
        vol_model = arch_model(resid, vol='GARCH', mean='zero'
                               , p=best_params[0]
                               , o=best_params[1]
                               , q=best_params[2],
                               dist=best_params[3])
        vol_res = vol_model.fit(disp='off',options={'maxiter':1000})
        if summary:
            print(vol_res.summary())
        cond_vol = vol_res.conditional_volatility
        vol_resid = vol_res.resid
        vol_std_resid = vol_resid / cond_vol
        var_quantile = vol_std_resid.quantile([.01, .05])
        resid_autocorr = self.autocorr_test(cond_vol, name + 'cond_vol')
        resid_stationary = self.stationarity_test(cond_vol, name + 'cond_vol')
        self.histogram(cond_vol, name + 'cond_vol')
        self.plot_autocorr(cond_vol, name + 'cond_vol')
        return cond_vol, var_quantile, [resid_autocorr, resid_stationary]

    def compute_empirical_VaR(self, fittedvalues, cond_vol, var_quantile):
        var_1 = - fittedvalues.values - cond_vol.values * var_quantile.values[0]
        var_5 = - fittedvalues.values - cond_vol.values * var_quantile.values[1]
        var_df = pd.DataFrame()
        var_df['0.01%'] = var_1
        var_df['0.05%'] = var_5
        return var_df

    def make_VaR_df(self, data1, data2, target_col, col_names):
        var_df_index = data1.index if len(data1) <= len(data2) else data2.index
        var_df = pd.concat([data1[target_col], data2[target_col]], axis=1).dropna()
        var_df.columns = col_names
        var_df.index = var_df_index
        return var_df

    def make_cond_vol_df(self, data1, data2, col_names):
        vol_df_index = data1.index if len(data1) <= len(data2) else data2.index
        vol_df = pd.DataFrame()
        vol_df[col_names[0]] = data1
        vol_df[col_names[1]] = data2
        vol_df.index = vol_df_index
        return vol_df

    def VAR(self, df, maxlags=1, ic='bic', summary=True):
        model = sm.tsa.VAR(df)
        var_res = model.fit(maxlags=maxlags, ic=ic) # trend = 'c'
        if summary:
            print(var_res.summary())
        self.VAR_lags.append([self.period, var_res.coefs.shape[0]])
        return var_res

    def granger_causality_test(self, var_res, caused, causing):
        gc_res = var_res.test_causality(caused, [causing])
        result = f"P Value of {causing} -> {caused} : {gc_res.pvalue}"
        return result, gc_res.pvalue

    def DY_spillover_index(self, var_res, mode, col_names, target=None, f_horizon=10, visualize=True):
        sigma_u = var_res.sigma_u.values
        sd_u = np.sqrt(np.diag(sigma_u))
        fevd = var_res.fevd(f_horizon, sigma_u / sd_u)
        if visualize:
            fevd.plot()
            plt.savefig(f'result/{self.period}/{mode}_fevd_plot.png')
            plt.close()
        fe = fevd.decomp[:, target, :]
        fevd_normal = (fe / fe.sum(axis=-1)[..., np.newaxis]) * 100
        fevd_normal = fevd_normal.swapaxes(0, 1)
        contrib_to = np.array([sub.sum(axis=0) - np.diag(sub) for sub in fevd_normal])
        contrib_from = np.array([sub.sum(axis=1) - np.diag(sub) for sub in fevd_normal])
        net_spillover = contrib_to - contrib_from
        to_df = pd.DataFrame(contrib_to, columns=col_names, index=target)
        from_df = pd.DataFrame(contrib_from, columns=col_names, index=target)
        net_df = pd.DataFrame(net_spillover, columns=col_names, index=target)
        to_df.to_csv(f'result/{self.period}/{mode}_to_df.csv')
        from_df.to_csv(f'result/{self.period}/{mode}_from_df.csv')
        net_df.to_csv(f'result/{self.period}/{mode}_net_spillover.csv')
        print(net_df)
        return net_df

    def modeling(self, names, mu_params, resid_params):
        data1, data2 = self.set_subdata()
        rtn1, rtn2 = self.compute_return(data1, data2)
        eda_result = self.eda(rtn1, rtn2, names)
        mu_fittedvalues1, mu_resid1, resid_result1 = self.mu_modeling(rtn1, names[0], summary=False, **mu_params)
        mu_fittedvalues2, mu_resid2, resid_result2 = self.mu_modeling(rtn2, names[1], summary=False, **mu_params)
        cond_vol1, var_quantile1, cond_vol_result1 = self.resid_modeling(mu_resid1, **resid_params, name =names[0], summary=False)
        cond_vol2, var_quantile2, cond_vol_result2 = self.resid_modeling(mu_resid2, **resid_params, name =names[1], summary=False)
        self.mu_fittedvalues = [mu_fittedvalues1, mu_fittedvalues2]
        self.cond_vols = [cond_vol1, cond_vol2]
        self.var_quantiles = [var_quantile1, var_quantile2]
        return eda_result

    def run(self, mode, names, VAR_maxlags=1, VAR_ic='bic', fevd_target=None, fevd_horizon=10):
        assert mode in ['var05', 'var01', 'vol'], "type은 ['var05', 'var01', 'vol'] 만 지원"
        if mode.startswith('var'):
            var_df1 = self.compute_empirical_VaR(self.mu_fittedvalues[0], self.cond_vols[0], self.var_quantiles[0])
            var_df2 = self.compute_empirical_VaR(self.mu_fittedvalues[1], self.cond_vols[1], self.var_quantiles[0])
        if mode == 'var05':
            df = self.make_VaR_df(var_df1, var_df2, '0.05%', names)
        elif mode == 'var01':
            df = self.make_VaR_df(var_df1, var_df2, '0.01%', names)
        else:
            df = self.make_cond_vol_df(self.cond_vols[0], self.cond_vols[1], names)
        VAR_res = self.VAR(df, maxlags=VAR_maxlags, ic=VAR_ic, summary=False)
        test_res1, pvalue1 = self.granger_causality_test(VAR_res, names[0], names[1])
        test_res2, pvalue2 = self.granger_causality_test(VAR_res, names[1], names[0])
        test_res = test_res1 + '\n' + test_res2
        with open(f'result/{self.period}/{mode}_granger_causality_test.txt', 'w') as f:
            f.write(test_res)
        net_df = self.DY_spillover_index(VAR_res, mode, col_names=names, target=fevd_target, f_horizon=fevd_horizon,
                                         visualize=True)
        return net_df, pvalue1, pvalue2
