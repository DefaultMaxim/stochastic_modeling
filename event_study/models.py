import numpy as np
from portfolio_models import Markowitz_portfolio
from dataclasses import dataclass

"""
    Model contains utils to run modeling of returns
    there are parameters:
    estimation_size: int (default 120),
    event_window_size: int (default 10),
"""

@dataclass
class Return_models:
    
    """
    Return event study models for calculate abnormal returns.
    """
    
    market: np.ndarray
    stock: np.ndarray
    risk_free: float = 0.1
    
    def market_model(
        self,
        est_t0: int,
        est_size: int,
        event_size: int
    ) -> np.ndarray:
        
        """
        Calculates abnormal returns based on market model.
        
        Parameters:
        - est_t0 (int): Starting estimation time index.
        - est_size (int): Estimation window size.
        - event_size (int): Event window size.

        Returns:
        np.ndarray: Abnormal returns.
            
        Note:
        Needs full stock price vector and full market arr[arr]. 
        !!!If there is only estimation + event windows NEEDS est_t0 = 0!!!
        """
        

        t1 = est_t0 + est_size
        t2 = t1 + 1 + event_size
        
        est_window = [i for i in range(est_t0, t1)] # for example [3, 4, 5, 6]
        event_window = [i for i in range(t1, t2)] # for example [7, 8, 9, 10]
        
        X = self.market[est_window]
        y = self.stock[est_window]
        
        market_returns = np.sum(X[est_window], axis=0)
        
        # может быть степень больше, надо подумать над этим!
        b, a = np.polyfit(X, y, deg=1) # linear regression coefs
        
        returns = a + b*market_returns[event_window] + \
            np.random.normal(0, 1, size=len(market_returns[event_window]))
        
        self.abnormal_returns = self.stock[event_window] - returns
        
        return self

    def constant_mean(returns,
                    estimation_size: int = 120,
                    event_window_size: int = 10,
                    full_ar: bool = False):
        """
        Model which calculate abnormal returns by $R - \mathbb{E}(R_{t}$
        :param returns: stock returns wo market returns
        :param estimation_size: estimation_size
        :param event_window_size: event_window_size
        :param full_ar: bool: True or False: returns ar on whole estimation period for some tests
        :return: AR, df, var, mean
        """
        mean = np.mean(returns[:estimation_size])

        res = np.array(returns) - mean

        df = estimation_size - 1

        variance = np.var(res)

        if full_ar:

            return res, df, variance, mean

        return res[-event_window_size:], df, variance, mean
