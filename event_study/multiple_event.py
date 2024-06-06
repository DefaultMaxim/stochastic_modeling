import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from IPython.display import display
import pandas as pd
from utils import get_all_attributes, get_horizons, market_plot
pd.set_option('display.max_columns', None)


def multiple_t_test(portfolio,
                    event_date,
                    alpha: float = 0.05):

    """
    Calculates t_stat, p_value, for each day in event window for portfolio
    :param portfolio: close price portfolio with market price
    :param event_date: event date
    :param alpha: significance level
    :return:

    """

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio, event_date=event_date)

    aar = 1/len(ar) * np.sum(ar, axis=0)

    car = np.cumsum(ar, axis=1)

    caar = np.cumsum(aar)

    var_aar = np.array([(1/len(ar)**2) * np.sum([x.var() for x in ar], axis=0)] * len(ar[0]))

    var_caar = np.array([np.sum(var_aar[:i]) for i in range(1, len(ar[0]) + 1)])

    t_stat = caar/np.sqrt(var_caar)

    df = np.sum(df)

    p_value = (1.0 - t.cdf(abs(t_stat), df=df))*2

    sign = t.ppf(1 - alpha/2, df=df)

    data = {'AAR': aar,
            'CAAR': caar,
            'var_AAR': var_aar,
            'var_CAAR': var_caar,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': abs(t_stat) > sign,
            'idx': np.arange(-len(aar) // 2, len(aar) // 2)}

    res = pd.DataFrame(data)

    return res


def multiple_cross_sectional_test(portfolio,
                                  event_date,
                                  alpha: float = 0.05,
                                  method: str = 'aar'.upper()):
    """

    https://www.eventstudytools.com/significance-tests#Csect
    Cross-Sectional Test (Abbr.: CSect T)
    :param portfolio: portfolio with returns
    :param method: string: aar or caar
    :param event_date: event date
    :param alpha: float: significance level
    :return: residual table

    """

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio, event_date=event_date)

    if method.upper() == 'aar'.upper():

        aar = 1 / len(ar) * np.sum(ar, axis=0)

        arr = []

        for j in range(len(aar)):

            arr.append(np.array([(ar[i][j] - aar[j]) ** 2 for i in range(len(ar))]))

        s_aar = np.sqrt(1/(len(ar) - 1) * np.array([np.sum(arr[i]) for i in range(len(arr))]))

        t_stat = np.sqrt(len(ar)) * aar/s_aar

        p_value = (1.0 - t.cdf(abs(t_stat), df=len(ar) - 1))

        sign = t.ppf(1 - alpha/2, df=len(ar) - 1)

        data = {'AAR': aar,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': abs(t_stat) > sign,
                'idx': np.arange(-len(aar) // 2, len(aar) // 2)}

        res = pd.DataFrame(data=data, index=ar[0].index)

        return res

    elif method.upper() == 'caar'.upper():

        car = np.sum(ar, axis=1)

        caar = 1/len(ar) * np.sum(car)

        s_caar = np.sqrt(1/len(ar) * np.sum((car - caar)**2))

        t_stat = np.sqrt(len(ar)) * caar/s_caar

        p_value = (1.0 - t.cdf(abs(t_stat), df=len(ar)-1))*2

        sign = t.ppf(1 - alpha/2, df=len(ar) - 1)

        data = {#'car': car,
                #'aar': aar,
                'caar': caar,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': abs(t_stat) > sign,
                #'idx': np.arange(-len(caar) // 2, len(caar) // 2)
            }

        res = pd.DataFrame(data=data, index=ar[0].index)

        return res

    else:

        return 'method does not exist'


def crude_dependence(portfolio,
                     event_date,
                     alpha: float = 0.05,
                     method: str = 'aar'):

    """

    Time-Series Standard Deviation or Crude Dependence Test (Abbr.: CDA T)
    https://www.eventstudytools.com/significance-tests#CDA
    :param portfolio: portfolio with market price
    :param event_date: event date
    :param alpha: significance level
    :param method: aar or caar method
    :return: residual table

    """

    event_idx, horizon_idx, start_idx, end_idx = get_horizons(portfolio, event_date)

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio,
                                                 event_date=event_date,
                                                 model='lin_reg',
                                                 full_ar=True)

    n = (end_idx - horizon_idx) // 2

    for i in range(len(ar)):

        ar[i] = ar[i][start_idx:event_idx + n]

    aar = 1 / len(ar) * np.sum(ar, axis=0)

    if method.upper() == 'aar'.upper():

        s_aar = np.sqrt(1/(len(aar) - 1) * np.sum((aar - 1/len(aar) * np.sum(aar))**2))

        t_stat = np.sqrt(len(ar)) * aar/s_aar

        p_value = (1.0 - t.cdf(abs(t_stat), df=len(aar) - 1))*2

        sign = t.ppf(1 - alpha/2, df=len(aar) - 1)

        data = {'aar': aar,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': abs(t_stat) > sign}

        res = pd.DataFrame(data=data, index=ar[0].index)

        return res[-2*n:]

    elif method.upper() == 'caar'.upper():

        car = np.sum(ar, axis=1)

        caar = np.cumsum(aar)

        #caar = np.sum(car)

        s_caar = np.sqrt(1/(len(aar) - 1) * np.sum((caar - 1/len(aar) * np.sum(caar))**2))

        t_stat = np.sqrt(len(ar)) * caar/s_caar

        p_value = (1.0 - t.cdf(abs(t_stat), df=len(aar) - 1)) * 2

        sign = t.ppf(1 - alpha / 2, df=len(aar) - 1)

        data = {'aar': aar,
                'caar': caar,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': abs(t_stat) > sign}

        res = pd.DataFrame(data=data, index=ar[0].index)

        return res[-2 * n:]

    else:

        return 'method does not exist'


def skewness_corrected_test(portfolio,
                            event_date,
                            alpha: float = 0.05,
                            method: str = 'aar'):

    """

    https://www.eventstudytools.com/significance-tests#skewness
    :param portfolio: portfolio with market price
    :param event_date: event date
    :param alpha: significance level
    :param method: aar or caar method
    :return: residual table

    """

    event_idx, horizon_idx, start_idx, end_idx = get_horizons(portfolio, event_date)

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio,
                                                 event_date=event_date,
                                                 model='lin_reg',
                                                 full_ar=True)

    n = (end_idx - horizon_idx) // 2

    for i in range(len(ar)):

        ar[i] = ar[i][start_idx:event_idx + n]

    aar = 1/len(ar) * np.sum(ar, axis=0)

    arr = []

    for j in range(len(aar)):

        arr.append(np.array([(ar[i][j] - aar[j]) ** 2 for i in range(len(ar))]))

    arr = np.array(arr)

    s_aar = np.sqrt(1 / (len(ar) - 1) * np.array([np.sum(arr[i]) for i in range(len(arr))]))

    tmp_arr = [((ar[i][-n] - aar[-n]) ** 3) / (s_aar[-n] ** 3) for i in range(len(ar))]

    gamma = len(ar)/((len(ar) - 2)*(len(ar) - 1)) * np.sum(tmp_arr)

    s = aar / s_aar

    t_stat = np.sqrt(len(ar)) * (s + 1/3 * gamma * s**2 + 1/27 * gamma**2 * s**3 + 1/(6*len(ar))*gamma)

    t_stat = t_stat

    p_value = (1.0 - t.cdf(abs(t_stat), df=len(ar) - 1)) * 2

    sign = t.ppf(1 - alpha/2, df=len(ar) - 1)

    data = {'aar': aar,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': abs(t_stat) > sign}

    res = pd.DataFrame(data=data, index=ar[0].index)

    return res[-2 * n:]

#
# close = pd.read_csv('data_import/datasets/close_portfolio.csv', index_col='time')
#
# close['market'] = [close.iloc[i].sum() for i in range(len(close))]
#
# close.index = pd.to_datetime(close.index)
#
# date = pd.to_datetime('2022-02-24 07:00:00+00:00')
#
# result_1 = multiple_t_test(portfolio=close, event_date=date)
#
# result_2_aar = multiple_cross_sectional_test(portfolio=close, event_date=date, method='aar')
# result_2_caar = multiple_cross_sectional_test(portfolio=close, event_date=date, method='caar')
#
# result_3_aar = crude_dependence(portfolio=close, event_date=date, method='aar')
# result_3_caar = crude_dependence(portfolio=close, event_date=date, method='caar')
#
# result_4_aar = skewness_corrected_test(portfolio=close, event_date=date, method='aar')
#
# print('RES 1 T_TEST')
# display(result_1)
#
# print('RES 2 CROSS AAR')
# display(result_2_aar)
#
# print('RES 2 CROSS CAAR')
# display(result_2_caar)
#
# print('RES 3 CRUDE AAR')
# display(result_3_aar)
#
# print('RES 3 CRUDE CAAR')
# display(result_3_caar)
#
# print('RES 4 SKEW AAR')
# display(result_4_aar)
# result_4_aar.to_csv('res_example.csv')
#
# market_plot(close.market, event_date=date)
