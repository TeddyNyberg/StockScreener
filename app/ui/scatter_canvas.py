import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from app.data.data_cache import get_yfdata_cache



class MplCanvas(FigureCanvas):
    def __init__(self, fig, parent=None):
        self.figure = fig
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


def create_return_figure(nyberg_ticker, ticker2 = "SPY"):
    df_nyberg, df_2 = get_yfdata_cache([nyberg_ticker, ticker2], "1Y")

    nyberg_returns = df_nyberg['Close'].pct_change().mul(100)
    spy_returns = df_2['Close'].pct_change().mul(100)

    returns_df = pd.DataFrame({
        'Nyberg_Return': nyberg_returns,
        'SPY_Return': spy_returns
    }).dropna()

    X = returns_df['Nyberg_Return']
    Y = returns_df['SPY_Return']

    plt.style.use('seaborn-v0_8-darkgrid')
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    max_abs_return = returns_df[['Nyberg_Return', 'SPY_Return']].abs().max().max() * 1.05
    plot_limit = max(0.1, max_abs_return)

    m, b = np.polyfit(X,Y,1)
    r = X.corr(Y)

    # Scatter plot: Nyberg (X) vs. SPY (Y)
    ax.scatter(
        X,
        Y,
        s=70,
        color='#1f77b4',
        alpha=0.6,
        edgecolors='white',
        linewidths=0.8
    )

    ax.plot([-plot_limit, plot_limit], [-plot_limit, plot_limit],
            'r--',
            alpha=0.5,
            label='Y = X (Equal Returns)'
            )

    """
    ax.plot(X, m * X + b,
            color='#d62728',
            linestyle='-',
            linewidth=3,
            label=rf'Trendline ($\beta$={m:.2f}, R={r:.2f})',
            zorder=3
            )
    """
    """
    ax.plot(X.mean(), Y.mean(),
            marker='X',
            markersize=14,
            markeredgewidth=2.5,
            markeredgecolor='white',
            markerfacecolor='#2ca02c',
            linestyle='',
            label=rf'Center of Gravity ({X.mean():.2f}, {Y.mean():.2f})',
            zorder=4
            )
    """
    ax.plot(gmean(X), gmean(Y),
            marker='X',
            markersize=14,
            markeredgewidth=2.5,
            markeredgecolor='white',
            markerfacecolor='#2ca02c',
            linestyle='',
            label=rf'Geometric mean ({gmean(X):.2f}, {gmean(Y):.2f})',
            zorder=4
            )

    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

    ax.set_title(f'Volatility and Correlation: {nyberg_ticker} vs. {ticker2} Daily % Returns', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{nyberg_ticker} Daily Return (%)', fontsize=14)
    ax.set_ylabel(f'{ticker2} Daily Return (%)', fontsize=14)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    ax.legend(loc='lower right')
    fig.tight_layout()

    return fig


def gmean(x):
    x = x+100
    a = np.log(x)
    return np.exp(a.mean()) - 100


