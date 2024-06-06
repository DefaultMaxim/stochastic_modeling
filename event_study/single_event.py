import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from IPython.display import display
import pandas as pd
from utils import get_all_attributes, portfolio_plot, get_horizons


def single_t_test(portfolio, event_date, alpha: float = 0.05):

    """

    Calculates t_stat, p_value, for each day in event window for each stock in portfolio

    table example:
                                     AR    ||   t_stat  ||      p_value  ||  significant ||  idx
    time
    2022-02-17 07:00:00+00:00 || -0.004615 || -0.282736 ||  7.778703e-01 ||  False       ||  -4
    2022-02-18 07:00:00+00:00 || -0.005193 || -0.318181 ||  7.509051e-01 ||  False       ||  -3
    2022-02-21 07:00:00+00:00 || -0.092273 || -5.653522 ||  1.097126e-07 ||   True       ||  -2
    2022-02-22 07:00:00+00:00 ||  0.036081 ||  2.210656 ||  2.897153e-02 ||   True       ||  -1
    2022-02-24 07:00:00+00:00 || -0.052915 || -3.242099 ||  1.539530e-03 ||   True       ||   0
    2022-02-25 07:00:00+00:00 || -0.040583 || -2.486503 ||  1.428745e-02 ||   True       ||   1
    2022-03-24 07:00:00+00:00 || -0.024659 || -1.510843 ||  1.334794e-01 ||  False       ||   2
    2022-03-25 07:00:00+00:00 || -0.001697 || -0.103983 ||  9.173582e-01 ||  False       ||   3
    2022-03-28 07:00:00+00:00 ||  0.037194 ||  2.278854 ||  2.445861e-02 ||   True       ||   4
    2022-03-29 07:00:00+00:00 ||  0.026116 ||  1.600091 ||  1.122300e-01 ||  False       ||   5

    :param portfolio: close price portfolio with market returns
    :param event_date: date event
    :param alpha: significance level (default 0.05)
    :return: array of residual tables

    """

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio, event_date=event_date)

    t_stat = [ar[i] / std[i] for i in range(len(ar))]

    p_value = [(1.0 - t.cdf(abs(t_stat[i]), df=df[i])) * 2 for i in range(len(t_stat))]

    sign = [t.ppf(1 - alpha / 2, df=df[i]) for i in range(len(df))]

    data = [{'AR': ar[i].values,
             't_stat': t_stat[i],
             'p_value': p_value[i],
             'significant': abs(t_stat[i]) > sign[i],
             'idx': np.arange(-len(ar[i]) // 2, len(ar[i]) // 2)}
            for i in range(len(ar))]

    res = [pd.DataFrame(data=data[i], index=ar[0].index) for i in range(len(ar))]

    return res


def single_criterion_t_test(portfolio, event_date, alpha: float = .05):

    """

    https://www.eventstudytools.com/significance-tests#A_1
    returns like single_t_test
    :param portfolio: close price portfolio with market returns
    :param event_date: event_date
    :param alpha: significance level
    :return: list of residuals tables

    """

    event_idx, horizon_idx, start_idx, end_idx = get_horizons(portfolio, event_date)

    ar, df, var, std, model = get_all_attributes(portfolio=portfolio,
                                                 event_date=event_date,
                                                 full_ar=True)

    n = (end_idx - horizon_idx) // 2

    for i in range(len(ar)):

        ar[i] = ar[i][start_idx:event_idx + n]

    s_ar = np.sqrt(1/(df[0] - 2) * np.sum(np.array(ar)**2, axis=1))

    for i in range(len(ar)):

        ar[i] = ar[i][-2 * n:]

    t_stat = np.array([ar[i]/s_ar[i] for i in range(len(ar))])

    p_value = np.round((1.0 - t.cdf(abs(t_stat), df=df[0] - 2))*2, 3)

    sign = [t.ppf(1 - alpha / 2, df=df[i] - 2) for i in range(len(df))]

    data = [{'AR': ar[i].values,
             't_stat': t_stat[i],
             'p_value': p_value[i],
             'significant': abs(t_stat[i]) > sign[i],
             'idx': np.arange(-len(ar[i]) // 2, len(ar[i]) // 2)}
            for i in range(len(ar))]

    res = [pd.DataFrame(data=data[i], index=ar[0].index) for i in range(len(ar))]

    return res


def permutation_test(portfolio):

    pass

#
# close = pd.read_csv('data_import/datasets/close_portfolio.csv', index_col='time')
#
# close['market'] = [close.iloc[i].sum() for i in range(len(close))]
#
# close.index = pd.to_datetime(close.index)
#
# date = pd.to_datetime('2022-02-24 07:00:00+00:00')
#
# results_first = single_t_test(portfolio=close, event_date=date)
#
# results_second = single_criterion_t_test(portfolio=close, event_date=date)
#
# display(close)
#
# for i in range(len(results_first)):
#
#     display(i, close.columns[i])
#
#     display(results_first[i])
#
#     display(results_second[i])
#
# portfolio_plot(portfolio=close, event_date=date)

