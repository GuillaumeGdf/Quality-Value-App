from functools import reduce
from typing import Callable

import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tikr_2016_2024 import Analysis, calculate_historical_cagr, monte_carlo_sensitivity_analysis


def make_descriptive_regression(results_list: list[pd.Series]):
    # Création d'un DataFrame consolidé de tous les scores (notre matrice A pour l'entraînement)
    common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in results_list))
    results_list = [df.loc[common_index] for df in results_list]

    if len(results_list) == 6:
        all_factors_train = pd.DataFrame({
            'Revenue_Rank': results_list[0]['Rank'],
            'GrossMargin_Rank': results_list[1]['Rank'],
            'FCF_Rank': results_list[2]['Rank'],
            'ROIC_Rank': results_list[3]['Rank'],
            'Buyback_Rank': results_list[4]['Rank'],
            'Debt_Rank': results_list[5]
        }, index=common_index)

        # Calcul de la variable cible : CAGR 2016-2022
        print(f"Calcul de la variable cible (CAGR {TRAIN_START}-{TRAIN_END})...")
        price_stats = calculate_historical_cagr(common_index, start_date=TRAIN_START, end_date=TRAIN_END)
        cagr = price_stats['CAGR']  # On ne garde que la colonne CAGR

        # Merge pour n'avoir que les actions avec des données complètes
        all_data_train = all_factors_train.join(cagr, how='inner').dropna()
        X = all_data_train[
            ['Revenue_Rank', 'GrossMargin_Rank', 'FCF_Rank', 'ROIC_Rank', 'Buyback_Rank', 'Debt_Rank']]
        y = all_data_train['CAGR']

    elif len(results_list) == 3:
        all_factors_train = pd.DataFrame({
            'Robust_Quality_Rank': results_list[0],
            'Value_Rank': results_list[1]['Rank'],
            'Piotroski_Rank': results_list[2]['Rank'],
        }, index=common_index)

        # Calcul de la variable cible : CAGR 2016-2022
        print(f"Calcul de la variable cible (CAGR {TEST_START}-{TEST_END})...")
        price_stats = calculate_historical_cagr(common_index, start_date=TEST_START, end_date=TEST_END)
        cagr = price_stats['CAGR']  # On ne garde que la colonne CAGR

        # Merge pour n'avoir que les actions avec des données complètes
        all_data_train = all_factors_train.join(cagr, how='inner').dropna()
        X = all_data_train[
            ['Robust_Quality_Rank', 'Value_Rank', 'Piotroski_Rank']]
        y = all_data_train['CAGR']

    else:
        all_factors_train = pd.DataFrame({
            'Revenue_Rank': results_list[0]['Rank'],
            'GrossMargin_Rank': results_list[1]['Rank'],
            'FCF_Rank': results_list[2]['Rank'],
            'ROIC_Rank': results_list[3]['Rank'],
            'Buyback_Rank': results_list[4]['Rank'],
            'Debt_Rank': results_list[5],
            'Value_Rank': results_list[6]['Rank'],
            'Piotroski_Rank': results_list[7]['Rank']
        }, index=common_index)

        # Calcul de la variable cible : CAGR 2016-2022
        print(f"Calcul de la variable cible (CAGR {TRAIN_START}-{TRAIN_END})...")
        price_stats = calculate_historical_cagr(common_index, start_date=TRAIN_START, end_date=TRAIN_END)
        cagr = price_stats['CAGR']  # On ne garde que la colonne CAGR

        # Merge pour n'avoir que les actions avec des données complètes
        all_data_train = all_factors_train.join(cagr, how='inner').dropna()
        X = all_data_train[
            ['Revenue_Rank', 'GrossMargin_Rank', 'FCF_Rank', 'ROIC_Rank', 'Buyback_Rank', 'Debt_Rank',
             'Value_Rank', 'Piotroski_Rank']]
        y = all_data_train['CAGR']

    # Entraînement du modèle de régression
    print("Entraînement du modèle de régression...")
    X_sm = sm.add_constant(X)  # ajoute intercept
    model_sm = sm.OLS(y, X_sm).fit()

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    #
    # # Evaluation in-sample
    # y_pred_train = model.predict(X_train)
    # r2_train = r2_score(y_train, y_pred_train)
    # print(f"R² en période d'entraînement (2016-2022) : {r2_train:.3f}")

    return model_sm, price_stats

# 1. ==================== CONFIGURATION ====================
# Chemins
root = Path(__file__).parent
filepath = root / "US Value 2016-2022.xlsx"

# Périodes de backtest
start_year = 2016
end_year = 2022

TRAIN_START = f"{start_year}-01-01"
TRAIN_END = f"{end_year}-12-31"  # Fin de la période d'entraînement
TEST_START = "2023-01-01" # Début de la période de test
TEST_END = "2025-07-01"   # Fin de la période de test (ajustable)

# 2. ==================== PHASE D'ENTRAÎNEMENT (2016-2022) ====================
print("🔧 PHASE D'ENTRAÎNEMENT (2016-2022)")
print("Chargement des données pour la période d'entraînement...")

# On initialise l'analyse avec TOUTES les données
an_full = Analysis(filepath, start_year, end_year)

# Mais on va filtrer pour ne calculer les scores que sur la sous-période
# Cette partie nécessiterait d'adapter votre classe Analysis pour accepter des dates en paramètre.
# Pour aujourd'hui, on suppose que vos fonctions calculent déjà sur 2016-2024.
# On va donc EXTRAPOLER en utilisant les scores 2016-2024 comme proxy pour 2016-2022.
# NOTE : C'est une approximation. L'idéal serait de modifier votre classe pour qu'elle puisse
# calculer les scores sur une période arbitraire.

'''
    Récupération des scores 
'''
print("Calcul des scores de facteurs (Quality, Value, Piotroski)...")
revenues_results = an_full.check_revenues_growth_and_stability()
opi_margin_results = an_full.check_gross_margin_growth_and_stability()
cfoa_results, groc_results = an_full.check_fcf_and_roic_gr()
buyback_results = an_full.check_shares_buyback()
value_results, _, _ = an_full.find_cheapest_stocks(risk_fre_rate=0.045)
piotroski_results, piotroski_results_reduced = an_full.compute_piotroski_score()

if f"{end_year}" != 2024 :
    debt_ebit = an_full.data_general[f'Total Debt {end_year}'] / an_full.data_general[f'EBIT {end_year}']
else:
    debt_ebit = an_full.data_general['Total Debt / EBITDA']

debt_series = (debt_ebit.rank(pct=True, ascending=False) * 100).sort_values(ascending=False)

'''
    Réalisation de la régression 
'''

train_dfs = [revenues_results, opi_margin_results, cfoa_results, groc_results, buyback_results, debt_series]
train_model, train_price_stats = make_descriptive_regression(results_list=train_dfs)

# 3. ==================== PHASE DE TEST (2023-2025) ====================
print("\n🔮 PHASE DE TEST (2023-2025)")
test_dfs = [revenues_results, opi_margin_results, cfoa_results, groc_results, buyback_results, debt_series,
       value_results, piotroski_results]

test_model, test_price_stats = make_descriptive_regression(results_list=test_dfs)

# Calcul du CAGR / Volatilité / Sharpe Ratio du SP500
train_benchmark_stats = calculate_historical_cagr(['SPY'], start_date=TRAIN_START, end_date=TRAIN_END)
test_benchmark_stats = calculate_historical_cagr(['SPY'], start_date=TEST_START, end_date=TEST_END)

#############
sel_freq_threshold = 0.40
# sel_freq_threshold = 2/3
sensitivity_hard_threshold = True

# Pondération du modèle factoriel
value_weight = 0.4

revenues_weight = 0.05
gross_margin_weight = 0.15
roic_weight = 0.2
debt_weight = 0.05
buyback_weight = 0.05
piotroski_weight = 0.1

quality_weight = revenues_weight + gross_margin_weight + roic_weight + debt_weight + buyback_weight

weights_sum = revenues_weight + gross_margin_weight + roic_weight + debt_weight + buyback_weight

revenues_weight_new = 0.05 / weights_sum
gross_margin_weight_new = 0.15 / weights_sum
roic_weight_new = 0.2 / weights_sum
debt_weight_new = 0.05 / weights_sum
buyback_weight_new = 0.05 / weights_sum
piotroski_weight_new = 0.1 / weights_sum

quality_results = (revenues_weight_new * train_dfs[0]['Rank'] +
                   gross_margin_weight_new * train_dfs[1]['Rank'] +
                   roic_weight_new * (train_dfs[2]['Rank'] + train_dfs[3]['Rank']) / 2 +
                   buyback_weight_new * train_dfs[4]['Rank'] +
                   debt_weight_new * train_dfs[5]
                   )
quality_results = quality_results.sort_values(ascending=False)
quality_results.name = 'Final Quality Ranking'

sensitivity_results = monte_carlo_sensitivity_analysis(an_full, num_simulations=2500)

# Pondération du classement Quality par la fréquence de sélection dans le top X% après N simulations
above_threshold_sel_freq = list(sensitivity_results[sensitivity_results['selection_frequency'] > sel_freq_threshold].index)

if sensitivity_hard_threshold:
    robust_quality_results = quality_results.loc[quality_results.index.isin(above_threshold_sel_freq)]
else:
    robust_quality_results = quality_results * sensitivity_results['Rank'] * 1e-2
    robust_quality_results = robust_quality_results[sensitivity_results['selection_frequency'] != 0]

robust_quality_results = robust_quality_results.sort_values(ascending=False)
robust_quality_list = list(robust_quality_results.index)

# Test en triant le classement Quality
other_list = list(quality_results[quality_results > 2/3*1e2].index)
test_dfs_robust = [value_results, quality_results.loc[other_list], piotroski_results]
common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in test_dfs_robust))
dfs = [df.loc[common_index] for df in test_dfs_robust]
final_list = value_weight * dfs[0]['Rank'] + quality_weight * dfs[1] + piotroski_weight * dfs[2]['Rank']
price_stats_test1 = calculate_historical_cagr(list(final_list.iloc[:20].index), start_date=TEST_START, end_date='2024-01-01')
print(f"Résultats test 1 : {price_stats_test1.mean(axis=0)}")

# Test en ne triant pas le classement Quality
test_dfs = [value_results, quality_results, piotroski_results_reduced]
common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in test_dfs))
dfs = [df.loc[common_index] for df in test_dfs]
final_list = value_weight * dfs[0]['Rank'] + quality_weight * dfs[1] + piotroski_weight * dfs[2]['Rank']
price_stats_test2 = calculate_historical_cagr(list(final_list.iloc[:50].index), start_date=TEST_START, end_date='2024-01-01')
print(f"Résultats test 2 : {price_stats_test1.mean(axis=0)}")
print(f"Résultats test 2 (20 meilleurs) : {price_stats_test1.iloc[:20].mean(axis=0)}")

# Test avec seulement les classements individuels en test
piotroski_only = calculate_historical_cagr(list(piotroski_results.index), start_date=TEST_START, end_date='2024-01-01')
value_only = calculate_historical_cagr(list(value_results.index), start_date=TEST_START, end_date='2024-01-01')
quality_only = calculate_historical_cagr(list(quality_results.index), start_date=TEST_START, end_date='2024-01-01')

# test_dfs_robust = [serie[serie.index.isin(robust_quality_list)] for serie in test_dfs]
# price_stats = calculate_historical_cagr(robust_quality_list, start_date=TEST_START, end_date='2024-01-01')

test_dfs_robust = [serie[serie.index.isin(other_list)] for serie in test_dfs]
price_stats = calculate_historical_cagr(other_list, start_date=TEST_START, end_date='2024-01-01')
test_benchmark_stats_1y = calculate_historical_cagr(['SPY'], start_date=TEST_START, end_date='2024-01-01')

seuil=0.1*len(price_stats)

common_index = reduce(lambda x, y: x.intersection(y), (df.index for df in test_dfs_robust))
dfs = [df.loc[common_index] for df in test_dfs_robust]

test_model_robust, test_price_stats_robust = make_descriptive_regression(results_list=test_dfs_robust)

print(f'Résultats pour les {seuil} meilleures actions')
print('='*50)
print(f'CAGR : {price_stats["CAGR"].iloc[:int(seuil)].mean()}')
print(f'Volatilité : {price_stats["Volatility"].iloc[:int(seuil)].mean()}')
print(f'Ratio de Sharpe : {price_stats["Sharpe Ratio"].iloc[:int(seuil)].mean()}')
print('-'*50)
print(f'CAGR : {test_benchmark_stats_1y["CAGR"]}')
print(f'Volatilité : {test_benchmark_stats_1y["Volatility"]}')
print(f'Ratio de Sharpe : {test_benchmark_stats_1y["Sharpe Ratio"]}')
print('='*50)

# 4. ==================== ANALYSE DES RÉSULTATS ====================
print("\n📊 ANALYSE DES RÉSULTATS HORS-ÉCHANTILLON")
print("Résultats de la régression sur les données d'entrainement")
print('-'*50)
print(train_model.summary())
print("Résultats de la régression sur les données de test")
print('-'*50)
print(test_model.summary())
print("Résultats de la régression sur les données de test (version Robuste)")
print('-'*50)
print(test_model_robust.summary())

# Correlation entre prédiction et réalité
# correlation = all_data_test['CAGR_Pred_2023_2025'].corr(all_data_test['CAGR_Real'])
# print(f"Corrélation Prédiction vs. Réalité (2023-2025) : {correlation:.3f}")
#
# # 5. ==================== CONCLUSION ====================
# print("\n" + "="*50)
# if correlation > 0:
#     print("🎉 Résultats encourageants ! Le modèle montre un pouvoir prédictif hors-échantillon.")
#     print("   La prochaine étape est d'affiner le calcul des facteurs pour la période de test.")
# else:
#     print("🤔 Le modèle n'a pas montré de pouvoir prédictif sur cette période.")
#     print("   Causes possibles : Période de test trop courte, changements de régime marché,")
#     print("   ou besoin d'affiner les facteurs.")

# Affichage des moyennes
print("Statistiques benchmark - Entrainement")
print(train_benchmark_stats)
print("Statistiques à poids égaux - Entrainement")
print(f"{train_price_stats['CAGR'].mean(), train_price_stats['Volatility'].mean(), train_price_stats['Sharpe Ratio'].mean()}")
print("Statistiques benchmark - Test")
print(test_benchmark_stats)
print("Statistiques à poids égaux - Test")
print(f"{test_price_stats['CAGR'].mean(), test_price_stats['Volatility'].mean(), test_price_stats['Sharpe Ratio'].mean()}")

print("="*50)