import numpy as np
import yfinance as yf
import pandas as pd


class PortfolioMetricsEngine:
    def __init__(self, tickers_list: list[str], start_date: str, end_date: str):
        self.tickers = tickers_list
        self.start_date = start_date
        self.end_date = end_date

        self.database: pd.DataFrame | None = None
        self.benchmark_metrics: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

        self.download_data()

    def download_data(self):
        self.database = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date, interval='1d')['Close']

    def compute_metrics(self, capital: float, stock_weights: np.ndarray, risk_free_rate: float = 0.045):
        if not np.isclose(stock_weights.sum(), 1.0):
            raise ValueError("La somme des poids du portefeuille doit être égale à 1.")

        # Calcul du montant alloué à chaque action et nombre d'actions
        amount_by_stock = capital * stock_weights
        initial_prices = self.database.iloc[0]
        n_stocks = np.floor(amount_by_stock / initial_prices)

        # Valorisation initiale du portefeuille
        position_values = self.database * n_stocks.values
        portfolio_value = position_values.sum(axis=1)

        # Rendements quotidiens
        daily_returns = self.database.pct_change().dropna()
        daily_log_returns = np.log(self.database / self.database.shift(1)).dropna()

        weights_effective = (n_stocks * initial_prices) / (n_stocks * initial_prices).sum()

        # Volatilité via matrice de covariance
        cov_matrix = daily_returns.cov()
        volatility = np.sqrt(weights_effective.T @ cov_matrix.values @ weights_effective) * np.sqrt(252)

        # Rendements du portefeuille
        portfolio_returns = daily_returns.dot(weights_effective)
        portfolio_log_returns = daily_log_returns.dot(weights_effective)

        # Rendement annualisé
        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1

        # Maximum drawdown
        peak = portfolio_value.cummax()
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = drawdown.max()

        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility

        # Stockage des métriques
        self.metrics = {
            "Final_Capital": portfolio_value.iloc[-1],
            "Total_Return": total_return,
            "Annualized_Return": annualized_return,
            "Volatility": volatility,
            "Portfolio_Value_Series": portfolio_value,
            "Portfolio_Returns": portfolio_returns,
            "Portfolio_LogReturns": portfolio_log_returns,
            "Drawdown_Series": drawdown,
            "Max_Drawdown": max_drawdown,
            "Sharpe_Ratio": sharpe_ratio
        }

    def compute_benchmark_metrics(self, is_US: bool = True, risk_free_rate: float = 0.045):
        if is_US:
            data = yf.download(['SPY'], self.start_date, self.end_date, interval='1d')['Close']
            data = data.rename(columns={'SPY': 'Close'})
        else:
            return

        # Calcul des rendements simples quotidiens
        data['Daily_Returns'] = data['Close'].pct_change().dropna()

        # Calcul du rendement total
        total_return = (data['Close'].iloc[-1] / data['Close'][0]) - 1

        # Calcul du rendement annualisé
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1

        # Calcul de la volatilité annualisée
        volatility = data['Daily_Returns'].std() * np.sqrt(252)

        # Calcul du drawdown maximal
        cumulative_returns = (1 + data['Daily_Returns']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = drawdown.min()

        # Calcul du Sharpe ratio annualisé
        excess_returns = data['Daily_Returns'] - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Compilation des résultats
        self.benchmark_metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }


a = PortfolioMetricsEngine(tickers_list=['AAPL', 'NVDA', 'MSFT'], start_date='2020-01-01', end_date='2024-01-01')
r = a.download_data()
c = a.compute_metrics(capital=5000, stock_weights=np.array([0.3, 0.4, 0.3]))

print()
