from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import norm


class RobustFinancialLoader:
    def __init__(self, dfs: list, mode: str = 'Z-score', feature_names: list = None):
        """
        Générateur de la matrice des variables explicatives (X).

        Args:
            dfs: Liste de DataFrames contenant les features. Chaque DF doit
                 avoir la colonne indexée par le Ticker.
            mode: 'Z-score' (cible 'Score_Final' dans [-5,5]) ou
                  'Rank' (cible 'Rank' dans [0, 100]).
            feature_names: Liste optionnelle de noms pour les variables
                           (ex: ['Growth', 'ROIC', 'Value', ...]).
        """
        self.dfs = dfs
        self.mode = mode.lower()  # Tolérance sur la casse
        if self.mode not in ['z-score', 'rank']:
            raise ValueError("[ERREUR] Le mode doit être 'Z-score' ou 'Rank'.")

        self.input_feature_names = feature_names
        self.data = None
        self.final_feature_names = None

        self._build_dataset()

    # =========================
    # 1. Sélection des features utiles
    # =========================
    def _select_features(self, df: pd.DataFrame, feature_name: str):
        # Détermination de la colonne cible
        target_col = 'Score_Final' if self.mode == 'z-score' else 'Rank'

        if target_col not in df.columns:
            raise KeyError(f"[ERREUR] La colonne '{target_col}' est absente du DataFrame pour {feature_name}.")

        # On isole uniquement la métrique ciblée
        sub = df[[target_col]].copy()

        # On renomme la colonne avec le nom de la feature pour la matrice X
        sub.columns = [feature_name]

        return sub

    # =========================
    # 2. Alignement des entreprises
    # =========================
    def _align_dataframes(self):
        dfs_clean = []

        for i, df in enumerate(self.dfs):
            # Attribution d'un nom explicite si fourni, sinon F0, F1...
            name = self.input_feature_names[i] if self.input_feature_names else f"Feature_{i}"
            sub = self._select_features(df, name)
            dfs_clean.append(sub)

        # Jointure sur intersection des indices (Tickers)
        data = dfs_clean[0]
        for df in dfs_clean[1:]:
            data = data.join(df, how="inner")

        return data

    # =========================
    # 3. Nettoyage
    # =========================
    def _clean_data(self, data):
        # Suppression inf / NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        return data

    # =========================
    # 4. Vérification homogénéité
    # =========================
    def _check_scaling(self, data):
        """
        Vérifie que les variables respectent l'échelle du mode choisi.
        """
        max_expected = 5.1 if self.mode == 'z-score' else 100.1

        for col in data.columns:
            actual_max = data[col].abs().max()
            if actual_max > max_expected:
                print(f"[WARNING] La feature '{col}' dépasse l'échelle prévue. "
                      f"(Max attendu: ~{max_expected}, Actuel: {actual_max:.2f})")

        return data

    # =========================
    # 5. Option : dé-corrélation simple
    # =========================
    def _remove_high_corr(self, data, threshold=0.9):
        """
        Pour les Z-score, corrélation par Pearson.
        Pour les ranking en percentiles, corrélation de Spearman.

        Args:
            data:
            threshold:

        Returns:

        """
        corr = data.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if to_drop:
            print(f"[INFO] Features supprimées pour colinéarité (> {threshold}) : {to_drop}")

        return data.drop(columns=to_drop)

    def _check_condition_number(self, data: pd.DataFrame):
        data = data.values
        cond = np.linalg.cond(data)
        print(f"Conditionnement: {cond:.4f}")

    # =========================
    # 6. Construction finale
    # =========================
    def _build_dataset(self):
        data = self._align_dataframes()
        data = self._clean_data(data)
        data = self._check_scaling(data)
        self._check_condition_number(data)

        # Ajout de variables non linéaires
        data = self._add_interactions(data)

        # Réduction de corrélation
        data = self._remove_high_corr(data)
        self._check_condition_number(data)

        self.data = data
        self.final_feature_names = data.columns.tolist()

    def _add_interactions(self, data):
        """Optionnel : à appeler explicitement si besoin de polynômes de degré 2"""
        interactions = [('Revenues', 'Value'),
                         ('ROIC', 'Value'),
                         ('Revenues', 'CFOA')]

        for c1, c2 in interactions:
            if c1 in data.columns and c2 in data.columns:
                new_col = f"{c1}_x_{c2}"

                v1 = data[c1]
                v2 = data[c2]

                raw_prod = v1 * v2

                # On doit corriger le fait que 2 négatifs produisent encore un négatif.
                corrected_prod = raw_prod.where(~((v1 < 0) & (v2 < 0)), -raw_prod.abs())

                # Normalisation sinon nouvelles variables non homogènes au reste
                median = corrected_prod.median()
                mad = (corrected_prod - median).abs().median()
                if mad == 0:
                    mad = 1

                data[new_col] = (corrected_prod - median) / (mad + 1e-8)
                data[new_col] = data[new_col].clip(-5, 5)

        return data

    # =========================
    # 7. Outputs
    # =========================
    def get_dataframe(self):
        return self.data

    def get_X(self):
        return self.data.values

    def get_feature_names(self):
        return self.final_feature_names


class RobustTargetGenerator:
    def __init__(self, start_date, price_df: pd.DataFrame, sector_mapping: pd.Series):
        """
        Génère le vecteur cible y de façon robuste et neutralisée.

        Args:
            price_df: DataFrame des prix (Total Return) avec Tickers en colonnes et Dates en index.
            sector_mapping: Series avec Ticker en index et Nom du Secteur en valeur.
        """
        self.start_date = start_date
        self.prices = price_df
        self.sectors = sector_mapping

    def calculate_forward_returns(self, date_t, horizons: Optional[list[int]] = None):
        """
        Calcule la moyenne des rendements logarithmiques sur T+6, T+12, T+18 mois.
        """
        # Conversion de l'index en datetime si nécessaire
        if horizons is None:
            horizons = [6, 12, 18]

        self.prices.index = pd.to_datetime(self.prices.index)
        date_t = self.prices.index[self.prices.index.get_indexer([date_t], method='nearest')[0]]

        returns_list = []
        for h in horizons:
            # Calcul de la date future (approximation par 21 jours de bourse par mois)
            target_date = date_t + pd.DateOffset(months=h)

            # On cherche la date de bourse la plus proche disponible
            actual_target_date = self.prices.index[self.prices.index.get_indexer([target_date], method='nearest')[0]]

            # Rendement logarithmique : ln(P_future / P_now)
            # Note : On utilise .loc pour s'assurer de ne pas avoir de look-ahead bias
            ret = np.log(self.prices.loc[actual_target_date] / self.prices.loc[date_t])
            returns_list.append(ret)

        # Moyenne des rendements (T+6, T+12, T+18) / 3
        avg_return = pd.concat(returns_list, axis=1).mean(axis=1)
        return avg_return

    def get_robust_y(self, date_t, method='Option_B', risk_adjust=False):
        """
        Génère le vecteur Y final.

        Args:
            date_t: La date charnière (fin de la période X, début de la période Y).
            method: 'Option_A' (MAD Z-Score) ou 'Option_B' (Rank-Inverse Normal).
            risk_adjust: Si True, divise par la volatilité passée (Sharpe-like).
        """
        # 1. Calcul du rendement moyen futur
        y_raw = self.calculate_forward_returns(date_t)
        date_t = pd.to_datetime(date_t)
        # 2. Ajustement par le risque (Volatilité historique sur 12 mois avant date_t)
        if risk_adjust:
            start_vol = date_t - pd.DateOffset(years=1)
            past_returns = np.log(self.prices.loc[start_vol:date_t]).diff()
            vol = past_returns.std() * np.sqrt(252)
            y_raw = y_raw / vol

        # 3. Neutralisation sectorielle (Excess Return vs Sector Median)
        df = pd.DataFrame({'sector': self.sectors, 'y': y_raw})
        df = df.dropna()

        # Soustraire la médiane de chaque secteur
        df['y_excess'] = df.groupby('sector')['y'].transform(lambda x: x - x.median())

        # 4. Robustification finale
        if method == 'Option_A':
            # Option A : MAD Z-Score
            median = df['y_excess'].median()
            mad = (df['y_excess'] - median).abs().median()
            y_final = (df['y_excess'] - median) / (mad + 1e-8)
            y_final = y_final.clip(-5, 5)  # Clipping robuste

        elif method == 'Option_B':
            # Option B : Rank-based Inverse Normal Transformation (Gaussianisation)
            # On transforme les rangs en probabilités, puis on passe par la fonction quantile Normale
            ranks = df['y_excess'].rank(method='average')
            percentiles = ranks / (len(ranks) + 1)
            y_final = pd.Series(norm.ppf(percentiles), index=df.index)

        else:
            raise ValueError("Méthode inconnue. Choisir 'Option_A' ou 'Option_B'.")

        return df, y_final