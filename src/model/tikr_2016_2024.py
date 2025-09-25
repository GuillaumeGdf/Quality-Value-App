import datetime
import time

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from functools import reduce
from datetime import datetime
from scipy.stats import gmean, skew, kurtosis
from sklearn.linear_model import LinearRegression

# Dictionnaire créé à partir des données médianes de Damodaran : https://pages.stern.nyu.edu/~adamodar/
# Chaque grand secteur (ex: IT) est divisé en sous-secteurs et on prend la moyenne des valeurs médianes
EV_OV_EBIT_MEDIAN = {'Information Technology': 27.72,
                     'Communication Services': 13.72,
                     'Consumer Staples': 15.98,
                     'Healthcare': 23.37,
                     'Consumer Discretionary': 32.5,
                     'Industrials': 24.,
                     'Materials': 13.34,
                     'Utilities': 22.,
                     'Energy': 10.14
                     }

def write_excel(filepath: Path,
                dfs: list[pd.DataFrame],
                sheet_names: list[str]
                ):

    # Création du fichier Excel
    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        for df, sheet in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet, index=True)


class ExcelLoader:
    def __init__(self, filepath):
        """
        Initialise le chargeur avec le chemin du fichier Excel.

        :param filepath: Chemin vers le fichier Excel (.xlsx ou .xls)
        """
        self.filepath = filepath
        self.sheet_names = self._get_sheet_names()
        self.dataframes = {}

    def _get_sheet_names(self):
        """
        Retourne la liste des noms de feuilles dans le fichier Excel.
        """
        try:
            xls = pd.ExcelFile(self.filepath)
            return xls.sheet_names
        except Exception as e:
            print(f"Erreur lors de l'accès aux feuilles : {e}")
            return []

    def load_sheet(self, sheet_name=None, usecols=None, index_col=None, skiprows=None) -> pd.DataFrame:
        """
        Charge une feuille spécifique dans un DataFrame.

        :param sheet_name: Nom de la feuille (ou indice) à charger. Si None, charge la première feuille.
        :param usecols: Colonnes à charger (ex: "A:D" ou ["A", "C"]).
        :param index_col: Colonne à utiliser comme index (numéro ou nom).
        :param skiprows: Lignes à ignorer au début (entier ou liste).
        :return: DataFrame chargé.
        """
        try:
            df = pd.read_excel(
                self.filepath,
                sheet_name=sheet_name,
                usecols=usecols,
                index_col=index_col,
                skiprows=skiprows,
                engine='openpyxl',  # explicite pour .xlsx
                na_values=["NaN", "-", "n/a", "NA"]
            )
            key = sheet_name if sheet_name else self.sheet_names[0]
            self.dataframes[key] = df
            return df
        except Exception as e:
            print(f"Erreur lors du chargement de la feuille '{sheet_name}': {e}")
            return None

    def get_loaded_dataframes(self):
        """
        Retourne les DataFrames déjà chargés.
        """
        return self.dataframes

    def get_sheet_names(self):
        """
        Retourne à nouveau les noms de feuilles.
        """
        return self.sheet_names


class Analysis:
    def __init__(self, filepath: Path,
                 start_year: int = 2016,
                 end_year: int = 2024,
                 sheet_name1: str = 'General',
                 sheet_name2: str = 'Gross Margin',
                 sheet_name3: str = 'FCF & ROIC',
                 sheet_name4: str = 'Piotroski'
                 ):
        # Chargement du fichier EXCEL
        loader = ExcelLoader(filepath=filepath)

        self.start_year = start_year
        self.end_year = end_year

        # Chargement des données
        self.data_general: pd.DataFrame = loader.load_sheet(sheet_name1)
        self.data_gross_margin: pd.DataFrame = loader.load_sheet(sheet_name2)
        self.data_fcf_roic: pd.DataFrame = loader.load_sheet(sheet_name3)
        self.data_piotroski: pd.DataFrame = loader.load_sheet(sheet_name4)

        # Modification des index
        self.data_general = self.data_general.set_index('Ticker')
        self.data_gross_margin = self.data_gross_margin.set_index('Ticker')
        self.data_fcf_roic = self.data_fcf_roic.set_index('Ticker')
        self.data_piotroski = self.data_piotroski.set_index('Ticker')

        # Suppression des lignes inutiles
        self.data_gross_margin = self.data_gross_margin.reindex(self.data_general.index)
        self.data_fcf_roic = self.data_fcf_roic.reindex(self.data_general.index)
        self.data_piotroski = self.data_piotroski.reindex(self.data_general.index)

    def check_revenues_growth_and_stability(self,
                                            threshold_growth: float = 0.02,
                                            stability_weight: float = 0.8,
                                            growth_weight: float = 0.2,
                                            ) -> pd.DataFrame:
        """
        Calcule la moyenne géométrique et la stabilité des variations annuelles du chiffre d'affaires. On demande
        un minimum de 2% de croissance annualisée. Sinon, pénalisation de 20% sur le score. Le score final est une
        pondération de la croissance (20%) avec la stabilité (80%).

        Args:
            growth_weight: poids associé au classement de la croissance
            stability_weight: poids associé au classement de la stabilité
            threshold_growth: seuil minimum à satisfaire pour la croissance du CA

        Returns: dataframe contenant le classement final des entreprises ayant la meilleure combinaison
         (croissance, stabilité)

        """
        revenues_names = [k for k in self.data_gross_margin.columns if 'Total Revenues 20' in k]
        revenues_names = [name for name in revenues_names
                          if self.start_year <= int(name.split()[-1]) <= self.end_year
                          ]
        revenues = self.data_gross_margin[revenues_names]
        revenues = revenues.apply(pd.to_numeric, errors='coerce')
        revenues = revenues.dropna()

        # Calcul de la moyenne géométrique des taux de variations annuelles pour chaque entreprise
        def annual_growth_rates(s):
            return s[1:].values / s[:-1].values - 1

        def geometric_growth(s):
            rates = annual_growth_rates(s) + 1
            return gmean(rates) - 1

        def stability_ratio(s):
            rates = annual_growth_rates(s)
            mean_rate = np.mean(rates)
            std_rate = np.std(rates, ddof=1)  # ddof=1 pour l'échantillon
            if std_rate == 0:
                return np.nan  # éviter division par zéro
            return mean_rate / std_rate

        geometric_means = revenues.apply(geometric_growth, axis=1)
        stability = revenues.apply(stability_ratio, axis=1)

        res = pd.DataFrame({'Geometric_Mean': geometric_means,
                            'Stability': stability
                            }
                           )

        # Création des classements intermédiaires en fonction de la croissance et de la stabilité
        res['Rank_Pct_GM'] = res['Geometric_Mean'].rank(pct=True) * 100
        res['Rank_Pct_Stability'] = res['Stability'].rank(pct=True) * 100

        # On pénalise les entreprises ayant une croissance du CA inférieure à 2%.
        penalize = pd.Series(
            np.where(res['Geometric_Mean'] < threshold_growth, 0.8, 1.),
            index=res.index
        )

        # Classement final
        res['Rank'] = stability_weight * res['Rank_Pct_GM'] + growth_weight * res['Rank_Pct_Stability']
        res['Rank'] = res['Rank'] * penalize
        res = res.sort_values(by='Rank', ascending=False)

        return res

    def find_cheapest_stocks(self,
                             percentile: float = 0.8,
                             risk_fre_rate: float = 0.04
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compare l'EBIT/TEV des actions par-rapport à l'EBIT/TEV médian des secteurs associés. Un premier filtre est
        effectué en ne gardant que les actions dont l'EBIT/TEV est supérieur à la valeur médiane. Ensuite, on calcule
        la distance relative (par-rapport à la médiane) et, par secteur on estime que les actions sous-évaluées sont
        les 10% les plus éloignées.

        Returns: dataframe contenant les EBIT/TEV et al distance relative, un dataframe contenant les statistiques par
        secteur et un dataframe contenant les actions sous-évaluées
        """
        res = pd.DataFrame()
        date = f'EBIT {self.end_year}' if self.end_year != 2024 else 'EBIT LTM'
        res['EBIT/TEV'] = self.data_general[date] / self.data_general['TEV']
        res['Sector'] = self.data_general['Sector']
        res = res.dropna(subset=['Sector'])

        # Ajout de la série contenant les valeurs médiannes par grand secteur
        sectors_med = []
        for val in res['Sector']:
            for key, med in EV_OV_EBIT_MEDIAN.items():
                if val == key:
                    sectors_med.append(1 / med)

        res['EBIT/TEV_MEDIAN'] = sectors_med

        # Value Condition
        res['AboveMedian&10YYield'] = (res['EBIT/TEV'] > res['EBIT/TEV_MEDIAN']) & (res["EBIT/TEV"] > risk_fre_rate)

        # On ne garde que les entreprises dont le prix est inférieur à la médiane de leur secteur
        res = res.loc[res['AboveMedian&10YYield'] == True]

        # Calcul de la distance relative pour déterminer si signal d'achat ou pas
        res['Relative_Distance'] = (res['EBIT/TEV'] - res['EBIT/TEV_MEDIAN']) / res['EBIT/TEV_MEDIAN']

        # Classement globale en fonction de la distance relative
        res = res.sort_values(by='Relative_Distance', ascending=False)

        # Création du classement en percentiles
        res['Rank'] = res['Relative_Distance'].rank(pct=True, ascending=True) * 100

        # Ajout d'une interprétation de la valorisation en fonction de la distance relative
        choices = ["Anomalie", "Sous-évaluée", "Fair"]
        conditions = [res['Relative_Distance'] > 1,
                      (res['Relative_Distance'] > 0.5) & (res['Relative_Distance'] < 1),
                      res['Relative_Distance'] <= 0.5
                      ]

        res['Opinion'] = np.select(conditions, choices, default="Non évaluée")

        # Calcul des statistiques par secteur en fonction de la distance relative
        sector_stats = res.groupby('Sector')['Relative_Distance'].agg([
            ('Nombre_Actions', 'count'),  # Nombre d'actions par secteur
            (f'Seuil_{int(percentile * 1e2)}percentile', lambda x: x.quantile(percentile)),  # Seuil des X% supérieurs
            ('Moyenne', 'mean'),  # Moyenne des distances relatives
            ('Mediane', 'median')  # Médiane des distances relatives
        ]).reset_index()

        # Récupération des X% les plus sous-évaluées par secteur
        tmp = pd.merge(res, sector_stats[['Sector', f'Seuil_{int(percentile * 1e2)}percentile']],
                       on='Sector',
                       how='left'
                       ).set_index(res.index)
        undervalued_stocks = tmp[tmp['Relative_Distance'] > tmp[f'Seuil_{int(percentile * 1e2)}percentile']]

        # Tri
        res = res.sort_values(by='Rank', ascending=False)

        return res, sector_stats, undervalued_stocks

    def check_shares_buyback(self, max_threshold: float = 0.04) -> pd.DataFrame:
        """
        On enlève les actions ayant une dilution de capital moyenne annuelle trop importante (> 4%). Ensuite, on classe
        les actions en fonction du taux de rachat : plus le rachat est important (CAGR négatif minimum) et mieux c'est
        pour l'actionnaire.

        Args:
            max_threshold: seuil au-delà duquel on ne souhaite pas aller

        Returns: dataframe contenant le classement final des actions avec le meilleur rachat d'actions

        """
        # Premier filtre sur les entreprises qui diluent vraiment trop le capital (CAGR > 4%), on affine ensuite
        n_years = 8
        buyback_df = self.data_general[[f'# of shares {self.start_year}', f'# of shares {self.end_year}']]
        buyback_df = buyback_df.apply(pd.to_numeric, errors='coerce')
        buyback_df['CAGR'] = (buyback_df[f'# of shares {self.end_year}'] /
                              buyback_df[f'# of shares {self.start_year}']) ** (1 / n_years) - 1
        buyback_df['Valid'] = buyback_df['CAGR'].apply(lambda x: x < max_threshold)

        # On ne garde que les actions satisfaisant la contidition dilution < 4%
        buyback_df = buyback_df[buyback_df['Valid']]

        # Classe les actions en fonction de leur rachat d'actions + réarrangement
        # buyback_df['Rank'] = buyback_df['CAGR'].rank(ascending=True, method='min').astype(int)
        buyback_df['Rank'] = buyback_df['CAGR'].rank(pct=True, ascending=False) * 100
        buyback_df = buyback_df.sort_values(by='Rank', ascending=False)
        # buyback_df = buyback_df.sort_values(by='Rank', ascending=True)

        return buyback_df

    def check_gross_margin_growth_and_stability(self) -> pd.DataFrame:
        gross_margin_names = [k for k in self.data_gross_margin.columns if 'Gross Profit Margin 20' in k]
        gross_margin_names = [name for name in gross_margin_names
                              if self.start_year <= int(name.split()[-2]) <= self.end_year
                              ]
        gross_margin = self.data_gross_margin[gross_margin_names]
        gross_margin = gross_margin.apply(pd.to_numeric, errors='coerce')
        gross_margin = gross_margin.dropna()
        gross_margin = gross_margin * 1e-2

        # Calcul de la moyenne géométrique des taux de variations annuelles pour chaque entreprise
        def stability_ratio(s):
            mean_rate = np.mean(s)
            std_rate = np.std(s, ddof=1)  # ddof=1 pour l'échantillon
            if std_rate == 0:
                return np.nan  # éviter division par zéro
            return mean_rate / std_rate

        def compute_gm(s):
            rates = s[1:].values / s[:-1].values
            prod = np.prod(rates)
            gm = prod ** (1 / rates.shape[0]) - 1
            return gm

        # Calcul de la moyenne géométrique du niveau absolu de la marge brute
        absolute_geometric_means = gross_margin.apply(gmean, axis=1)

        # Calcul de la moyenne géométrique des variations annuelles de la marge brute
        growth_geometric_means = gross_margin.apply(compute_gm, axis=1)

        stability = gross_margin.apply(stability_ratio, axis=1)

        res = pd.DataFrame({'Absolute_Geometric_Mean': absolute_geometric_means,
                            'Growth_Geometric_Mean': growth_geometric_means,
                            'Stability': stability
                            }
                           )

        def softmax_score(df: pd.DataFrame,
                          cols: list[str],
                          w_absolute: float = 1/3,
                          w_growth: float = 1/3,
                          w_stability: float = 1/3,
                          alpha=4
                          ):
            w = np.array([w_absolute, w_growth, w_stability])  # poids égaux
            X = df[cols].to_numpy(dtype=float)
            score = (np.sum(w * (X ** alpha), axis=1)) ** (1 / alpha)
            return pd.Series(score, index=df.index, name=f'SoftMax_alpha{alpha}')

        # TODO : essayer soft-max(niveau absolu, croissance, résidu (stabilité)) -- la stabilité est modéremment
        #  corrélé au niveau absolu (corr=0.53). Donc, 47% n'est pas expliquée. Pour expliquer ce manque, on peut
        # réaliser une régression linéaire niveau - stabilité et considérer le résidu. Il s'agit de la part non
        # expliquée par le modèle.

        # Création du classement final en fonction du niveau absolu, de la croissance et de la stabilité
        # Utilisation d'une fonction de soft-max au lieu du max afin de lisser la sélection
        res['Rank_Pct_Gross_GM'] = res['Absolute_Geometric_Mean'].rank(pct=True) * 100
        res['Rank_Pct_Growth_GM'] = res['Growth_Geometric_Mean'].rank(pct=True) * 100
        res['Rank_Pct_Stability'] = res['Stability'].rank(pct=True) * 100
        res['Rank'] = res[['Rank_Pct_Growth_GM', 'Rank_Pct_Stability']].max(axis=1)

        # cols = ['Rank_Pct_Gross_GM', 'Rank_Pct_Growth_GM', 'Rank_Pct_Stability']
        # res['Rank'] = softmax_score(res, cols=cols, alpha=4)

        # Trier les actions selon le maximum des percentiles
        res = res.sort_values(by='Rank', ascending=False)

        return res

    def check_fcf_and_roic_gr(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcul la moyenne géométrique des Free Cash Flow (somme des 10 dernières années) sur Totals Assets
        (dernière année).

        Calcul la moyenne géométrique des ROIC et compare ce niveau à la stabilité. On prend le max(niveau, stabilité)
        afin de garder leur meilleur des deux dimensions.

        Returns: DataFrame associé au FCF/Total Assets et DataFrame associé au ROIC
        """
        # Partie FCF
        fcf_names = [k for k in self.data_fcf_roic.columns if 'FCF 20' in k]
        fcf_names = [name for name in fcf_names
                     if self.start_year <= int(name.split()[-1]) <= self.end_year
                     ]
        fcf = self.data_fcf_roic[fcf_names]
        fcf = fcf.apply(pd.to_numeric, errors='coerce')
        fcf = fcf.dropna()

        cfoa = fcf.sum(axis=1) / self.data_fcf_roic[f'Total Assets {self.end_year}']
        res_cfoa = pd.DataFrame({'CFOA': cfoa})

        res_cfoa['Rank'] = res_cfoa['CFOA'].rank(pct=True) * 100
        res_cfoa = res_cfoa.sort_values(by='Rank', ascending=False)

        # Partie ROIC
        roic_names = [k for k in self.data_fcf_roic.columns if 'ROIC 20' in k]
        roic_names = [name for name in roic_names
                      if self.start_year <= int(name.split()[-1]) <= self.end_year
                      ]
        roic = self.data_fcf_roic[roic_names] * 1e-2
        roic = roic.apply(pd.to_numeric, errors='coerce')
        roic = roic.dropna()

        def geometric_growth(s):
            return gmean(1 + s) - 1

        def stability_ratio(s):
            mean_rate = np.mean(s)
            std_rate = np.std(s, ddof=1)  # ddof=1 pour l'échantillon
            if std_rate == 0:
                return np.nan  # éviter division par zéro
            return mean_rate / std_rate

        geometric_means = roic.apply(geometric_growth, axis=1)
        stability = roic.apply(stability_ratio, axis=1)

        res_groic = pd.DataFrame({'Geometric_Mean': geometric_means,
                                  'Stability': stability
                                  }
                                 )

        # Création du classement final en fonction du niveau moyen (moyenne géométrique) et de la stabilité
        res_groic['Rank_Pct_GM'] = res_groic['Geometric_Mean'].rank(pct=True) * 100
        res_groic['Rank_Pct_Stability'] = res_groic['Stability'].rank(pct=True) * 100

        # On considère le max(niveau, stabilité) si on veut chopper les meilleurs dans leur dimension
        res_groic['Rank'] = res_groic[['Rank_Pct_GM', 'Rank_Pct_Stability']].max(axis=1)

        # Trier les actions selon le maximum des percentiles
        res_groic = res_groic.sort_values(by='Rank', ascending=False)

        return res_cfoa, res_groic

    def compute_piotroski_score(self, score_threshold: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        pd.set_option('future.no_silent_downcasting', True)

        final_score = pd.Series(np.zeros(len(self.data_piotroski.index), dtype=int), index=self.data_piotroski.index,
                                name='Final_Score')

        # Nettoyage de la donnée
        colonnes_numeriques = ['Total Revenues LTM',
                               'Operating Income 2016', 'Operating Income LTM',
                               f'Net Income {self.end_year-1}', f'Net Income {self.end_year}',
                               f'Total Assets {self.end_year-1}', f'Total Assets {self.end_year}',
                               f'FCF {self.end_year-1}', f'FCF {self.end_year}',
                               f'LT Debt {self.end_year-1}', f'LT Debt {self.end_year}',
                               f'Total Current Assets {self.end_year-1}', f'Total Current Assets {self.end_year}',
                               f'Total Current Liabilities {self.end_year-1}', f'Total Current Liabilities {self.end_year}',
                               f'Operating Income {self.end_year-1}', f'Operating Income {self.end_year}',
                               f'Total Revenues {self.end_year-1}', f'Total Revenues {self.end_year}',
                               f'Cash from Operations {self.end_year-1}', f'Cash from Operations {self.end_year}'
                               ]

        # Étape 2 : Nettoyage et conversion
        for col in colonnes_numeriques:
            self.data_piotroski[col] = pd.to_numeric(self.data_piotroski[col], errors='coerce')
            if self.data_piotroski[col].dropna().mod(1).eq(0).all():  # Vérifie si entier
                self.data_piotroski[col] = self.data_piotroski[col].astype('Int64')  # 'Int64' gère les NaN

        # Gestion des <NA> (NaN de Pandas) dans les colonnes d'émission/rachat d'actions
        self.data_piotroski[f'Issuance of Common Stocks {self.end_year}'] = self.data_piotroski[
            f'Issuance of Common Stocks {self.end_year}'].fillna(0)
        self.data_piotroski[f'Repurchase of Common Stocks {self.end_year}'] = self.data_piotroski[
            f'Repurchase of Common Stocks {self.end_year}'].fillna(0)

        # Gestion des NaN dans le LT Debt
        self.data_piotroski[f'LT Debt {self.end_year-1}'] = self.data_piotroski[f'LT Debt {self.end_year-1}'].fillna(1e4)
        self.data_piotroski[f'LT Debt {self.end_year}'] = self.data_piotroski[f'LT Debt {self.end_year}'].fillna(1e4)

        self.data_piotroski = self.data_piotroski.dropna()

        # Rentabilité
        ROA = self.data_piotroski[f'Net Income {self.end_year}'] / self.data_piotroski[f'Total Assets {self.end_year}']
        final_score += (ROA > 0).astype(int)

        FCF = self.data_piotroski[f'FCF {self.end_year}'] / self.data_piotroski[f'Total Assets {self.end_year}']
        final_score += (FCF > 0).astype(int)

        accrual = ((self.data_piotroski[f'Net Income {self.end_year}'] - self.data_piotroski[f'Cash from Operations {self.end_year}']) /
                   self.data_piotroski[f'Total Assets {self.end_year}'])
        final_score += (accrual > ROA).astype(int)

        # Diminution de l'Endettement
        delta_leverage = (self.data_piotroski[f'LT Debt {self.end_year-1}'] / self.data_piotroski[f'Total Assets {self.end_year-1}']
                          - self.data_piotroski[f'LT Debt {self.end_year}'] / self.data_piotroski[f'Total Assets {self.end_year}'])
        final_score += (delta_leverage > 0).astype(int)

        # Amélioration de la Liquidité
        delta_liquid = (self.data_piotroski[f'Total Current Assets {self.end_year-1}'] / self.data_piotroski[
            f'Total Current Liabilities {self.end_year-1}']
                        - self.data_piotroski[f'Total Current Assets {self.end_year}'] / self.data_piotroski[
                            f'Total Current Liabilities {self.end_year}'])
        final_score += (delta_liquid < 0).astype(int)

        # Rachat d'actions > Emission d'actions
        self.data_piotroski[f'Repurchase of Common Stocks {self.end_year}'] = self.data_piotroski[
            f'Repurchase of Common Stocks {self.end_year}'].replace('-', 0)
        self.data_piotroski[f'Issuance of Common Stocks {self.end_year}'] = self.data_piotroski[
            f'Issuance of Common Stocks {self.end_year}'].replace('-', 0)

        shares_variation = (abs(self.data_piotroski[f'Repurchase of Common Stocks {self.end_year}']) -
                            self.data_piotroski[f'Issuance of Common Stocks {self.end_year}'])
        final_score += (shares_variation > 0).astype(int)

        # Efficacité opérationnelle
        delta_margin = (self.data_piotroski[f'Operating Income {self.end_year}'] / self.data_piotroski[f'Total Revenues {self.end_year}'] -
                        self.data_piotroski[f'Operating Income {self.end_year-1}'] / self.data_piotroski[f'Total Revenues {self.end_year-1}'])
        final_score += (delta_margin > 0).astype(int)

        delta_turn = (self.data_piotroski[f'Total Revenues {self.end_year}'] / self.data_piotroski[f'Total Assets {self.end_year}'] -
                      self.data_piotroski[f'Total Revenues {self.end_year-1}'] / self.data_piotroski[f'Total Assets {self.end_year-1}'])
        final_score += (delta_turn > 0).astype(int)

        delta_ROA = ROA - self.data_piotroski[f'Net Income {self.end_year-1}'] / self.data_piotroski[f'Total Assets {self.end_year-1}']
        final_score += (delta_ROA > 0).astype(int)

        delta_FCF = FCF - self.data_piotroski[f'FCF {self.end_year-1}'] / self.data_piotroski[f'Total Assets {self.end_year-1}']
        final_score += (delta_FCF > 0).astype(int)

        rank = final_score.rank(pct=True) * 1e2
        rank.name = 'Rank'
        final_score.name = 'Piotroski Score'

        # Création du dataframe final
        final_score = pd.DataFrame({final_score.name: final_score,
                                    rank.name: rank
                                    }
                                   )

        final_score = final_score.sort_values(by='Rank', ascending=False)
        final_score_sorted = final_score[final_score['Piotroski Score'] >= score_threshold]

        return final_score, final_score_sorted


def monte_carlo_sensitivity_analysis(an, num_simulations=500, top_percentile=0.2):
    """
    Analyse de sensibilité par méthode Monte-Carlo des poids du modèle factoriel

    Args:
        an: Instance de la classe Analysis
        num_simulations: Nombre de simulations Monte-Carlo
        top_percentile: Percentile pour définir le top des actions (défaut: top 20%)

    Returns:
        DataFrame avec la fréquence de sélection de chaque entreprise
    """
    # Récupération des résultats individuels
    revenues_results = an.check_revenues_growth_and_stability()
    opi_margin_results = an.check_gross_margin_growth_and_stability()
    cfoa_results, groc_results = an.check_fcf_and_roic_gr()
    buyback_results = an.check_shares_buyback()
    basic_data_results, _, _ = an.find_cheapest_stocks(risk_fre_rate=0.045)

    debt_results = ((an.data_general['Total Debt / EBITDA'].rank(pct=True, ascending=False) * 100).
                    sort_values(ascending=False))

    # Mise en commun des index
    dfs = [revenues_results, opi_margin_results, cfoa_results, groc_results,
           buyback_results, debt_results]

    common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))
    dfs = [df.loc[common_index] for df in dfs]

    # Initialisation du compteur
    selection_count = pd.Series(0, index=common_index, name='selection_count')

    # Poids initiaux pour référence
    initial_weights = {
        'revenues': 0.05,
        'gross_margin': 0.15,
        'roic': 0.2,
        'buyback': 0.05,
        'debt': 0.05
    }

    w_sum = sum(initial_weights.values())

    for key, val in initial_weights.items():
        initial_weights[key] = val / w_sum

    print(f"Début de l'analyse Monte-Carlo avec {num_simulations} simulations...")

    for i in range(num_simulations):
        if (i + 1) % 100 == 0:
            print(f"Simulation {i + 1}/{num_simulations}")

        # Génération de poids aléatoires (distribution Dirichlet pour garantir somme=1)
        random_weights = np.random.dirichlet([5, 15, 20, 5, 5, 10])

        # Calcul du score avec poids aléatoires
        random_score = (
                random_weights[0] * dfs[0]['Rank'] +
                random_weights[1] * dfs[1]['Rank'] +
                random_weights[2] * (dfs[2]['Rank'] + dfs[3]['Rank']) / 2 +
                random_weights[3] * dfs[4]['Rank'] +
                random_weights[4] * dfs[5]
        )

        # Sélection du top percentile
        threshold = random_score.quantile(1 - top_percentile)
        top_stocks = random_score[random_score >= threshold].index

        # Mise à jour du compteur
        selection_count.loc[top_stocks] += 1

    # Calcul des fréquences
    selection_frequency = selection_count / num_simulations

    # Résultats avec les poids initiaux pour comparaison
    initial_score = (
            initial_weights['revenues'] * dfs[0]['Rank'] +
            initial_weights['gross_margin'] * dfs[1]['Rank'] +
            initial_weights['roic'] * (dfs[2]['Rank'] + dfs[3]['Rank']) / 2 +
            initial_weights['buyback'] * dfs[4]['Rank'] +
            initial_weights['debt'] * dfs[5]
    )

    initial_threshold = initial_score.quantile(1 - top_percentile)
    initial_top_stocks = initial_score[initial_score >= initial_threshold].index

    # Création du dataframe de résultats
    results_df = pd.DataFrame({
        'selection_frequency': selection_frequency,
        'initial_selection': [1 if stock in initial_top_stocks else 0
                              for stock in selection_frequency.index],
        'Rank': selection_frequency.rank(pct=True) * 100,
        'sector': an.data_general.loc[selection_frequency.index, 'Sector']
    })

    # Statistiques supplémentaires
    robust_stocks = results_df[results_df['selection_frequency'] >= 0.8]
    sensitive_stocks = results_df[results_df['selection_frequency'] <= 0.2]

    print(f"\nRésultats de l'analyse:")
    print(f"Actions robustes (fréquence ≥ 80%): {len(robust_stocks)}")
    print(f"Actions sensibles (fréquence ≤ 20%): {len(sensitive_stocks)}")
    print(f"Correspondance avec sélection initiale: "
          f"{results_df['initial_selection'].mean():.1%}")

    return results_df.sort_values('selection_frequency', ascending=False)


def calculate_historical_cagr(ticker_list: list[str],
                              start_date: str = "2016-01-01",
                              end_date: str = "2025-01-01",
                              risk_free_rate: float = 0.04) -> pd.DataFrame:
    """
    Calcule le CAGR, la volatilité annualisée et le Sharpe ratio pour une liste de tickers.
    """
    cagrs = []
    vols = []
    sharpes = []
    tickers = []

    results = {}

    for ticker in ticker_list:
        try:
            # Téléchargement des prix
            stock_data = yf.download(ticker, start=start_date, end=end_date)

            if len(stock_data) > 0:
                start_price = stock_data['Close'].iloc[0]
                end_price = stock_data['Close'].iloc[-1]

                # Durée en années
                years = (datetime.strptime(end_date, "%Y-%m-%d") -
                         datetime.strptime(start_date, "%Y-%m-%d")).days / 365.25

                # CAGR
                cagr = (end_price / start_price) ** (1 / years) - 1
                cagr = float(cagr)

                # Rendements quotidiens
                daily_returns = stock_data['Close'].pct_change().dropna()

                # Volatilité annualisée
                vol_annual = daily_returns.std() * np.sqrt(252)
                vol_annual = float(vol_annual)

                # Sharpe ratio
                sharpe = (cagr - risk_free_rate) / vol_annual if (
                        vol_annual > 0) else np.nan

                tickers.append(ticker)
                cagrs.append(cagr)
                vols.append(vol_annual)
                sharpes.append(sharpe)

        except Exception as e:
            print(f"Erreur avec {ticker}: {e}")
            continue

    res = pd.DataFrame({'CAGR': cagrs,
                        'Volatility': vols,
                        'Sharpe Ratio': sharpes},
                       index=tickers
                       )

    return res



def analyze_model_performance(quality_stocks: list[str],
                              robust_quality_stocks: list[str],
                              benchmark_ticker: str = "SPY"):
    """
    Compare la performance des actions quality versus un benchmark
    """
    # Calcul du CAGR des actions Quality, Robust Quality et Benchmark
    quality_df = calculate_historical_cagr(quality_stocks)
    robust_quality_df = quality_df.loc[robust_quality_stocks]
    benchmark_df = calculate_historical_cagr([benchmark_ticker])

    outperformance_quality = quality_df['CAGR'].mean() - benchmark_df['CAGR'][benchmark_ticker]
    outperformance_robust_quality = robust_quality_df['CAGR'].mean() - benchmark_df['CAGR'][benchmark_ticker]

    hit_ratio_quality = ((quality_df['CAGR'] > benchmark_df['CAGR'][benchmark_ticker]).sum() /
                         len(quality_df.index))
    hit_ratio_robust_quality = ((robust_quality_df['CAGR'] > benchmark_df['CAGR'][benchmark_ticker]).sum() /
                                len(robust_quality_df.index))

    return {
        'Quality': quality_df,
        'Robust_Quality': robust_quality_df,
        'Benchmark_CAGR': benchmark_df['CAGR'][benchmark_ticker],
        'Benchmark_Volatility': benchmark_df['Volatility'][benchmark_ticker],
        'Benchmark_SR': benchmark_df['Sharpe Ratio'][benchmark_ticker],
        'Quality_CAGR': quality_df['CAGR'].mean(),
        'Quality_Volatility': quality_df['Volatility'].mean(),
        'Quality_SR': quality_df['Sharpe Ratio'].mean(),
        'Robust_Quality_CAGR': robust_quality_df['CAGR'].mean(),
        'Robust_Quality_Volatility': robust_quality_df['Volatility'].mean(),
        'Robust_Quality_SR': robust_quality_df['Sharpe Ratio'].mean(),
        'Outperformance_Quality': outperformance_quality,
        'Outperformance_Robust_Quality': outperformance_robust_quality,
        'Hit_Ratio_Quality': hit_ratio_quality,
        'Hit_Ratio_Robust_Quality': hit_ratio_robust_quality,
        'Sharpe_Quality': quality_df['Sharpe Ratio'].mean(),
        'Sharpe_Robust_Quality': robust_quality_df['Sharpe Ratio'].mean(),
        'Quality_CAGR_Stats': {
            'mean': quality_df['CAGR'].mean(),
            'median': quality_df['CAGR'].median(),
            'std': quality_df['CAGR'].std(),
            'min': quality_df['CAGR'].min(),
            'max': quality_df['CAGR'].max()
        },
        'Robust_Quality_CAGR_Stats': {
            'mean': robust_quality_df['CAGR'].mean(),
            'median': robust_quality_df['CAGR'].median(),
            'std': robust_quality_df['CAGR'].std(),
            'min': robust_quality_df['CAGR'].min(),
            'max': robust_quality_df['CAGR'].max()
        }
    }


def plot_comparisons(results):
    quality_cagr = results['Quality']['CAGR']
    robust_cagr = results['Robust_Quality']['CAGR']

    # Calcul des métriques statistiques
    quality_skew = skew(quality_cagr)
    quality_kurt = kurtosis(quality_cagr)
    robust_skew = skew(robust_cagr)
    robust_kurt = kurtosis(robust_cagr)

    # Affichage détaillé dans la console
    print("=" * 50)
    print("ANALYSE STATISTIQUE")
    print("=" * 50)
    print(f"QUALITY (n={len(quality_cagr)})")
    print(f"CAGR Moyen: {quality_cagr.mean()*1e2:.2f}%")
    print(f"CAGR Médian: {quality_cagr.median()*1e2:.2f}%")
    print(f"Volatilité: {results['Quality']['Volatility'].mean()*1e2:.2f}%")
    print(f"Skewness: {quality_skew:.4f}")
    print(f"Kurtosis: {quality_kurt:.4f}")
    print(f"Écart-type: {quality_cagr.std():.4f}")
    print()
    print(f"ROBUST QUALITY (n={len(robust_cagr)})")
    print(f"CAGR Moyen: {robust_cagr.mean()*1e2:.2f}%")
    print(f"CAGR Médian: {robust_cagr.median()*1e2:.2f}%")
    print(f"Volatilité: {results['Robust_Quality']['Volatility'].mean()*1e2:.2f}%")
    print(f"Skewness: {robust_skew:.4f}")
    print(f"Kurtosis: {robust_kurt:.4f}")
    print(f"Écart-type: {robust_cagr.std():.4f}")

    # Histogramme
    plt.figure(figsize=(10,5))
    plt.hist(quality_cagr*1e2, bins=30, alpha=0.7, label=f"Quality (n={len(quality_cagr.index)})",
             color="blue", density=True, edgecolor='black', linewidth=0.5)
    plt.hist(robust_cagr*1e2, bins=30, alpha=0.7, label=f"Robust Quality (n={len(robust_cagr.index)})",
             color="red", density=True, edgecolor='black', linewidth=0.5)

    # Lignes des moyennes
    plt.axvline(quality_cagr.mean()*1e2, color="darkblue", linestyle="--",
                linewidth=2, label=f"Mean Quality: {quality_cagr.mean()*1e2:.2f}%"
                                   f"\nSkew: {quality_skew:.3f}"
                                   f"\nKurtosis: {quality_kurt:.3f}")
    plt.axvline(robust_cagr.mean()*1e2, color="darkred", linestyle="--",
                linewidth=2, label=f"Mean Robust: {robust_cagr.mean()*1e2:.2f}%"
                                   f"\nSkew: {robust_skew:.3f}"
                                   f"\nKurtosis: {robust_kurt:.3f}")

    plt.title("Distribution des CAGR - Quality vs Robust Quality")
    plt.xlabel("CAGR (%)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.show()

def main():
    ###################################################
    #                       Tests
    ###################################################
    current_file = Path(__file__).resolve()
    root = current_file.parent

    # Données d'entrée
    export_results = True
    monte_carlo_sim = True
    US_stocks = True

    top_quality_threshold = 0.1
    sel_freq_threshold = 0.50
    # sel_freq_threshold = 2/3
    sensitivity_hard_threshold = True

    # Taux sans risque actuel
    rfr = 0.04

    # Pondération du modèle factoriel
    value_weight = 0.45

    revenues_weight = 0.05
    gross_margin_weight = 0.15
    roic_weight = 0.15
    debt_weight = 0.05
    buyback_weight = 0.05
    piotroski_weight = 0.1

    quality_weight = revenues_weight + gross_margin_weight + roic_weight + debt_weight + buyback_weight

    # Projection des poids ci-dessus pour le classement purement Quality
    weights_sum = revenues_weight + gross_margin_weight + roic_weight + debt_weight + buyback_weight

    revenues_weight_new = 0.05 / weights_sum
    gross_margin_weight_new = 0.15 / weights_sum
    roic_weight_new = 0.2 / weights_sum
    debt_weight_new = 0.05 / weights_sum
    buyback_weight_new = 0.05 / weights_sum
    piotroski_weight_new = 0.1 / weights_sum

    #######################################
    #       Début des calculs...
    #######################################
    if US_stocks:
        filename = "US Value 2016-2024.xlsx"
    else:
        filename = "EU Value 2016-2024.xlsx"

    path = root / filename

    start = time.process_time()

    an = Analysis(filepath=path)
    revenues_results = an.check_revenues_growth_and_stability()
    gross_margin_results = an.check_gross_margin_growth_and_stability()
    cfoa_results, groc_results = an.check_fcf_and_roic_gr()
    piotroski_results, piotroski_results_reduced = an.compute_piotroski_score()

    buyback_results = an.check_shares_buyback()
    basic_data_results, sector_stats_results, undervalued_stocks_results = an.find_cheapest_stocks(risk_fre_rate=rfr)

    debt_results = (an.data_general['Total Debt / EBITDA'].rank(pct=True, ascending=False) * 100).sort_values(
        ascending=False)

    #######################################
    #         Classement Quality Only
    #######################################
    # Mise en commun des dataframes : certains tickers ont pu être supprimés
    dfs = [revenues_results, gross_margin_results, cfoa_results, groc_results, buyback_results, debt_results]

    # Trouver les index communs à tous les DataFrames
    common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))

    # Mettre à jour chaque DataFrame en ne gardant que ces index
    dfs = [df.loc[common_index] for df in dfs]

    # Création du classement final Quality (i.e. sans la valorisation EBIT/TEV).
    quality_results = (revenues_weight_new * dfs[0]['Rank'] +
                       gross_margin_weight_new * dfs[1]['Rank'] +
                       roic_weight_new * (dfs[2]['Rank'] + dfs[3]['Rank']) / 2 +
                       buyback_weight_new * dfs[4]['Rank'] +
                       debt_weight_new * dfs[5]
                       )
    quality_results = quality_results.sort_values(ascending=False)
    quality_results.name = 'Final Quality Ranking'

    # Analyse de sensibilité du classement Quality afin de s'assurer que l'on considère bien un top 20% robuste au biais
    # de la pondération arbitraire.
    sensitivity_results = monte_carlo_sensitivity_analysis(an, num_simulations=2500)

    # Pondération du classement Quality par la fréquence de sélection dans le top X% après N simulations
    above_threshold_sel_freq = list(sensitivity_results[sensitivity_results['selection_frequency'] > sel_freq_threshold].index)

    # Soit on prend toutes les actions au-dessus d'un certain seuil
    if sensitivity_hard_threshold:
        robust_quality_results = quality_results.loc[quality_results.index.isin(above_threshold_sel_freq)]
    else:
        robust_quality_results = quality_results * sensitivity_results['Rank'] * 1e-2
        robust_quality_results = robust_quality_results[sensitivity_results['selection_frequency'] != 0]

    robust_quality_results = robust_quality_results.sort_values(ascending=False)

    seuil = top_quality_threshold * len(quality_results)
    above_average_quality_list = list(quality_results.iloc[:int(seuil)].index)
    quality_check = sensitivity_results.loc[above_average_quality_list]['selection_frequency'].mean()
    robust_quality_results = quality_results[quality_results > 50]

    # Calcul des métriques associées à la performance boursière passée
    # performance = analyze_model_performance(quality_results.index.tolist(),
    #                                         robust_quality_results.index.tolist()
    #                                         )
    # plot_comparisons(performance)


    # Création du DataFrame contenant tous les classements
    all_ranks_df = pd.DataFrame({'Revenues': revenues_results['Rank'],
                                 'Gross Margin': gross_margin_results['Rank'],
                                 'CFOA': cfoa_results['Rank'],
                                 'GROC': groc_results['Rank'],
                                 'BUYBACK': buyback_results['Rank'],
                                 'DEBT': debt_results,
                                 'Piotroski': piotroski_results['Rank'],
                                 'Value': basic_data_results['Rank'],
                                 }
                                )

    def check_factorial_correlation(rank_df: pd.DataFrame):
        global common_index, df, mask

        # Test de corrélation des facteurs
        import seaborn as sns

        list_of_dfs = [
            revenues_results[['Rank']].rename(columns={'Rank': 'Revenues_Rank'}),
            gross_margin_results[['Rank']].rename(columns={'Rank': 'GrossMargin_Rank'}),
            cfoa_results[['Rank']].rename(columns={'Rank': 'FCF_Rank'}),
            groc_results[['Rank']].rename(columns={'Rank': 'ROIC_Rank'}),
            piotroski_results[['Rank']].rename(columns={'Rank': 'Piotroski_Rank'}),
            basic_data_results[['Rank']].rename(columns={'Rank': 'Value_Rank'}),
            buyback_results[['Rank']].rename(columns={'Rank': 'Buyback_Rank'}),
            debt_results.to_frame(name='Debt_Rank')
        ]

        # 3. Trouver l'index commun à tous les dataframes
        common_index = list_of_dfs[0].index
        for df in list_of_dfs[1:]:
            common_index = common_index.intersection(df.index)
        print(f"Nombre d'actions dans l'analyse de corrélation : {len(common_index)}")

        # 4. Fusionner tous les scores sur l'index commun
        all_scores = list_of_dfs[0].loc[common_index]
        for df in list_of_dfs[1:]:
            all_scores = all_scores.join(df.loc[common_index], how='inner')

        # 5. Calculer la matrice de corrélation des rangs (Méthode de Spearman)
        # La corrélation de Spearman est parfaite pour les rangs.
        correlation_matrix = all_scores.corr(method='spearman')
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Masque pour n'afficher que la moitié inférieure
        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    cmap='RdBu_r',  # Palette de couleurs bleu (négatif) -> blanc -> rouge (positif)
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={"shrink": .8},
                    annot_kws={"size": 9})
        plt.title('Corrélation des rangs entre tous les facteurs du modèle (Spearman)\n', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


    # check_factorial_correlation(all_ranks_df)

    # Si suffisament de données, résolution du système linéaire Ax = b afin d'étudier la significativité des facteurs
    # Quality où A est la matrice des facteurs colonnes et b le CAGR de chaque action.

    ##############################################################################
    #         Classement Value + Quality + Piotroski
    ##############################################################################
    # Mise en commun des dataframes : certains tickers ont pu être supprimés
    # dfs = [basic_data_results, robust_quality_results, piotroski_results_reduced]
    dfs = [basic_data_results, quality_results, piotroski_results_reduced]
    common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in dfs))
    dfs = [df.loc[common_index] for df in dfs]

    # Création du classement final (Value + Quality + Piotroski) basé sur le modèle factoriel ci-dessus.
    value_results = (value_weight * dfs[0]['Rank'] +
                     quality_weight * dfs[1] +
                     piotroski_weight * dfs[2]['Rank']
                     )
    value_results = value_results.sort_values(ascending=False)
    value_results.name = 'Factor Model Ranking'

    # Calcul des poids à allouer pour les 20 premières actions
    top_20_sum = sum(value_results.iloc[:20].values)
    weights_allocation = value_results.iloc[:20].values / top_20_sum * 1e2

    # Récupération des 20% meilleures actions seulement
    top_value_results_threshold = value_results.quantile(0.8)
    final_value_score_results = value_results[value_results > top_value_results_threshold]
    final_value_sector_results = an.data_general['Sector'].loc[final_value_score_results.index]

    final_value_score_results = pd.DataFrame({"Value Score": final_value_score_results,
                                              "Sector": final_value_sector_results})

    #########################################################
    # Ecriture des résultats dans un Excel pour visualisation
    #########################################################
    # Stockage dans des conteneurs
    df_list = [revenues_results,
               gross_margin_results,
               cfoa_results,
               groc_results,
               piotroski_results,
               buyback_results,
               basic_data_results,
               sector_stats_results,
               undervalued_stocks_results,
               quality_results,
               sensitivity_results,
               robust_quality_results,
               value_results,
               final_value_score_results,
               ]

    df_list_names = ['Revenues', 'Gross Profit', 'Free Cash Flow', 'Return On Capital', 'Piotroski Score',
                     'Shares Buyback', 'EBIT_ov_TEV (Full)', 'EBIT_ov_TEV (Sector)', 'EBIT_ov_TEV (Undervalued)',
                     'Quality Ranking', 'Quality Sensitivity Analysis', 'Robust Quality', 'Factor Model Ranking', 'Top Factor Model Ranking'
                     ]

    if export_results:
        today = datetime.today().strftime("%d_%m_%Y")
        filename = f"{today}_{'US' if US_stocks else 'EU'}_value_results.xlsx"

        write_excel(filepath=root / filename,
                    dfs=df_list,
                    sheet_names=df_list_names
                    )

    print(f'Temps de calcul total: {time.process_time() - start}s')
    print('')

if __name__ == '__main__':
    main()