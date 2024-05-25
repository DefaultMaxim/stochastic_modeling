import numpy as np
from scipy.optimize import minimize
from numpy.random import uniform
import warnings


class Markowitz_portfolio:
    
    """
    Markowitz portfolio optimization class includes analytical methods and optimization methods.
    """
    
    def __init__(self, 
                 market: np.array):
        """
        Market pretransform must be: market.bfill().to_numpy().T
        Market = np.array, stocks*number_obs!!! (for example 14 x 5270 )
        """
        self.market = market
        self.returns = np.nan_to_num(np.diff(np.asanyarray(self.market)), nan=0.)
        self.mean_returns = np.mean(self.returns, axis=1)
        self.cov_matrix = np.nan_to_num(np.cov(self.returns), nan=0.)
        self.alpha, self.beta, self.gamma, self.delta = self.compute_params()
    
    def compute_params(self):
          
        """
        Calculates parameters of market
        $$
        \alpha = J_n^T \cdot \mathbb{V} \cdot J_n^T
        \beta = J_n^T \cdot \mathbb{V}^{-1} \cdot r
        \gamma = r^T \cdot \mathbb{V}^{-1} \cdot r
        \delta = \alpha \gamma - \beta^{2}
        J_n^T - matrix ones (n x n)
        \mathbb{V} - returns covariance matrix
        $$
        
        Parameters:
        - market (np.ndarray): close price market without market price
        
        Returns: 
        alpha, beta, gamma, delta
        """

        r = self.mean_returns
        v = self.cov_matrix
        
        # logging.info(f'r = {r}, v = {v}')

        ones = np.ones_like(r)

        v_inv = np.linalg.inv(v)

        tmp = np.dot(ones.T, v_inv)
        alpha = np.dot(tmp, ones)

        tmp = np.dot(ones.T, v_inv)
        beta = np.dot(tmp, r)

        tmp = np.dot(r, v_inv)
        gamma = np.dot(tmp, r)

        delta = alpha * gamma - beta ** 2

        return alpha, beta, gamma, delta
    
    def min_risk(self, keep_shorts: bool = True):
        """
        Compute Markowitz market model
        stay keep_shorts = True, false part is not ready
        
        Parameters:
        - market (np.ndarray): market pct change
        - keep_shorts (bool): If True then result market may contains short sales
        
        Returns: 
        portfolio, risk, returns
        """

        if self.beta < 0:
            
            warnings.warn(f'Beta = {np.round(self.beta, 4)} < 0, analytical solution may not work correctly.')
        
        r = self.mean_returns
        v = self.cov_matrix

        ones = np.ones_like(r)

        v_inv = np.linalg.inv(v)

        if keep_shorts:

            x_tmp = 1 / self.alpha * v_inv
            x = np.dot(x_tmp, ones)
            tmp = np.dot(x.T, v)

            returns = self.beta/self.alpha

            risk = 1/np.sqrt(self.alpha)

            portfolio = x.T
            
            self.portfolio = portfolio
            self.risk = risk
            self.returns = returns

            return self

    def max_sharpe(self, keep_shorts: bool = True):
        """
        Calculates market which maximize sharpe ratio = r/sigma
        stay keep_shorts = True, false part is not ready
        
        Parameters:
        - market (np.ndarray): Close price market without market price
        - keep_shorts (bool): If true then market may contains short sales
        
        Returns: 
        portfolio, risk, returns
        """

        if self.beta < 0:
            
            warnings.warn(f'Beta = {np.round(self.beta, 4)} < 0, analytical solution may not work correctly.',)
            
        r = self.mean_returns
        v = self.cov_matrix

        ones = np.ones_like(r)

        v_inv = np.linalg.inv(v)
    
        if keep_shorts:

            x = 1/self.beta * v_inv @ r

            returns = self.gamma/self.beta

            risk = np.sqrt(self.gamma)/self.beta

            portfolio = x.T
            
            self.portfolio = portfolio
            self.risk = risk
            self.returns = returns

            return self
        
    def monte_carlo(self):
        """
        Monte carlo portfolio optimization
        """
        
        pass
    
    def numerical_opt(self, target: str = 'sr'): 
        """
        Maximize sharpe_rate or minimize risk
        
        Parameters:
        - target (str): for minimize Sharpe ratio input 'sr', for minimize risk input 'risk'
        
        Returns: 
        portfolio, risk, returns
        """
        if target.upper() not in ['sr'.upper(), 'risk'.upper()]:
            
            raise ValueError(f'target must be "sr" or "risk", not {target}')
            
        
        def get_ret_vol_sr(weights): 
            """ gets returns, risk, sharpe_rate """
            weights = np.array(weights)
            ret = weights.T @ self.mean_returns * 100
            # 100 - число дней работы биржи
            # че это за 100?????????????? Каких дней??? 
            # CHECK!!!
            vol = np.sqrt(weights.T @ self.cov_matrix @ weights * 100**2)
            sr = ret/vol 
            return np.array([ret, vol, sr])

        def neg_sharpe(weights): 
            """ max ~ -min"""
            return get_ret_vol_sr(weights)[2] * (-1)

        def volot(weights):
            """ gets risk """
            return get_ret_vol_sr(weights)[1]

        def check_sum(weights): 
            """ \sum x_i = 1 """
            return np.sum(weights) - 1
          
        cons = ({'type':'eq','fun':check_sum})

        bounds = [(0, 1) for i in range(len(self.returns))]

        init_guess = uniform(0, len(self.returns), size=len(self.returns))
        init_guess /= np.sum(init_guess)

        if target.upper() == 'sr'.upper():
            
            opt_sp = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
            
            ret_sp = get_ret_vol_sr(opt_sp.x)[0]/100
            vol_sp = get_ret_vol_sr(opt_sp.x)[1]/100
            
            sharpe_ratio = ret_sp/vol_sp
            
            self.portfolio = init_guess
            self.risk = vol_sp
            self.returns = ret_sp
            self.sharpe_ratio = sharpe_ratio
            
            return self

        elif target.upper() == 'risk'.upper():
            
            opt_sgm = minimize(volot, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

            ret_sgm = get_ret_vol_sr(opt_sgm.x)[0]/100
            vol_sgm = get_ret_vol_sr(opt_sgm.x)[1]/100
            
            sharpe_ratio = ret_sgm/vol_sgm
            
            self.portfolio = init_guess
            self.risk = vol_sgm
            self.returns = ret_sgm
            self.sharpe_ratio = sharpe_ratio
            
            return self


from math import exp
from typing import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MertonPortfolio:
    mu: float
    sigma: float
    r: float
    rho: float
    horizon: float
    gamma: float
    epsilon: float = 1e-6

    def excess(self) -> float:
        return self.mu - self.r

    def variance(self) -> float:
        return self.sigma * self.sigma

    def allocation(self) -> float:
        return self.excess() / (self.gamma * self.variance())

    def portfolio_return(self) -> float:
        return self.r + self.allocation() * self.excess()

    def nu(self) -> float:
        return (self.rho - (1 - self.gamma) * self.portfolio_return()) / \
            self.gamma

    def f(self, time: float) -> float:
        remaining: float = self.horizon - time
        nu = self.nu()
        if nu == 0:
            ret = remaining + self.epsilon
        else:
            ret = (1 + (nu * self.epsilon - 1) * exp(-nu * remaining)) / nu
        return ret

    def fractional_consumption_rate(self, time: float) -> float:
        return 1 / self.f(time)

    def wealth_growth_rate(self, time: float) -> float:
        return self.portfolio_return() - self.fractional_consumption_rate(time)

    def expected_wealth(self, time: float) -> float:
        base: float = exp(self.portfolio_return() * time)
        nu = self.nu()
        if nu == 0:
            ret = base * (1 - time / (self.horizon + self.epsilon))
        else:
            ret = base * (1 - (1 - exp(-nu * time)) /
                          (1 + (nu * self.epsilon - 1) *
                           exp(-nu * self.horizon)))
        return ret
