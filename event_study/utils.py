import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import lin_reg
import math


def get_horizons(portfolio,
                 event_date,
                 event_window: int = 10,
                 estimation_size: int = 120):

    """

    calculates horizons of forecast by regression
    :param event_date: date of event
    :param event_window: event windows days
    :param portfolio: portfolio with dates
    :param estimation_size: estimation size
    :return: event idx, horizon_idx, start_idx, end_idx

    """

    event_idx = portfolio.index.get_loc(event_date)

    n = event_window // 2

    horizon_idx = event_idx - n

    if horizon_idx > estimation_size:

        start_idx = horizon_idx - estimation_size
    else:

        start_idx = 0

    end_idx = event_idx + n + 1

    return event_idx, horizon_idx, start_idx, end_idx


def get_attributes_of_stock(market,
                            stock,
                            event_date,
                            model: str = 'lin_reg'.upper(),
                            full_ar: bool = False):

    """

    calculates ar, df, var, std, model for each stock in portfolio
    :param market: market returns
    :param stock: stock returns
    :param event_date: event date
    :param model: str: check models
    :param full_ar: bool: False, switch True if you need ar on full period
    :return: ar:abnormal returns,
    df:degrees of freedom,
    var:variance ar,
    std: Mean squared error of the residuals,
    model: model

    """

    event_idx, horizon_idx, start_idx, end_idx = get_horizons(market, event_date)

    if model.upper() == 'lin_reg'.upper():

        ar, df, var, std, models = lin_reg(market,
                                           stock,
                                           start=start_idx,
                                           horizon=horizon_idx,
                                           full_ar=full_ar)

        return ar, df, var, std, models


def get_all_attributes(portfolio,
                       event_date,
                       model: str = 'lin_reg'.upper(),
                       full_ar: bool = False):

    """

    gets all arrays ar[], df[], var[], std[], model[]
    :param portfolio: portfolio with market price
    :param event_date: event date
    :param model: str: check models
    :param full_ar: bool: False, switch True if you need ar on full period
    :return: ar:array(days * stocks), df:array(len(stocks)), var:array(len(stocks)), std:array(len(stocks)), model

    """

    returns = portfolio.pct_change().fillna(0)

    ar = [0] * (len(returns.columns) - 1)

    df = [0] * (len(returns.columns) - 1)

    var = [0] * (len(returns.columns) - 1)

    std = [0] * (len(returns.columns) - 1)

    models = [0] * (len(returns.columns) - 1)

    for i in range((len(returns.columns) - 1)):
        ar[i], df[i], var[i], std[i], models[i] = get_attributes_of_stock(
                market=returns.market,
                stock=returns.iloc[:, i],
                event_date=event_date,
                model=model,
                full_ar=full_ar)

    return ar, df, var, std, models


def market_plot(market, event_date):

    """
    plots market and event date
    :param market: market price
    :param event_date: event date
    :return: plot
    """

    plt.figure(figsize=(14, 8))

    plt.plot(market)

    plt.axvline(event_date)

    plt.show()


def portfolio_plot(portfolio, event_date):

    """

    Plots of security prices of portfolio
    :param portfolio: portfolio
    :param event_date: event date
    :return: plot

    """

    i, j = 0, 0

    plots_per_row = 2

    fig, axs = plt.subplots(math.ceil((len(portfolio.columns) - 1)/plots_per_row),
                            plots_per_row,
                            figsize=(80, 80))

    for col in portfolio.columns[:-1]:

        axs[i][j].plot(portfolio[col])

        axs[i][j].set_ylabel(col)

        axs[i][j].axvline(event_date)

        j += 1

        if j % plots_per_row == 0:

            i += 1

            j = 0

    plt.show()


def to_table(data):
    pass
