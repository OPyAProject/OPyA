#General
import pandas as pd
import numpy as np
import copy

#Pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import risk_models
from pypfopt import plotting

#Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['axes.grid'] = True
import seaborn as sns

data100=pd.read_csv('data/data100_09042022.csv',index_col=0,header=0,parse_dates=True)
#data100=data100.iloc[:,:4]

mu = mean_historical_return(data100)
S = CovarianceShrinkage(data100).ledoit_wolf()
ef=EfficientFrontier(mu,S)

fig, ax = plt.subplots()
ef_max_sharpe = copy.deepcopy(ef)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio

ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 50000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("img/ef_scatter.png", dpi=200)
plt.show()