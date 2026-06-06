import numpy as np


def compute_lambert_robust_bic(X, y, beta, intercept, delta=1.35):
    """
    Calcule le BIC Robuste basé sur la formulation de Lambert-Lacroix (2011).
    """
    n, p = X.shape
    res = y - (X @ beta + intercept)

    # Estimation robuste de l'échelle (MAD)
    # mad = np.median(np.abs(res - np.median(res)))
    # scale = max(1.4826 * mad, 1e-6)

    # Calcul de la perte de Huber (mise à l'échelle)
    # abs_res_scaled = np.abs(res) / scale
    abs_res_scaled = np.abs(res)
    huber_loss_array = np.where(
        abs_res_scaled <= delta,
        abs_res_scaled ** 2,
        2 * delta * abs_res_scaled - delta ** 2
    )
    # Somme des pertes pondérée par l'échelle (pour respecter la dimensionnalité)
    # L_H = scale * np.sum(huber_loss_array)
    L_H = np.sum(huber_loss_array)

    # Degrés de liberté : nombre de coefficients strictement non nuls
    k = np.sum(np.abs(beta) > 1e-5)

    # Formulation stricte de Lambert-Lacroix (2011) : log(Loss) + k * log(n) / (2n)
    # Note : On multiplie souvent l'ensemble par 2n pour retrouver l'échelle standard du BIC
    bic_val = np.log(L_H + 1e-8) + k * (np.log(n) / (2 * n))

    return bic_val, k


import matplotlib.pyplot as plt


class ADMMWarmStartPath:
    """
    Calcule le chemin de régularisation complet en utilisant le Warm-Start sur l'ADMM.
    """

    def __init__(self, n_lambdas=50, eps=1e-4, delta=1.35, rho=1.0):
        self.n_lambdas = n_lambdas
        self.eps = eps
        self.delta = delta
        self.rho = rho

        self.lambdas_ = None
        self.coefs_path_ = None
        self.bics_ = None
        self.best_lambda_ = None
        self.best_coef_ = None
        self.best_intercept_ = None

    def fit(self, X, y, penalty_weights=None):
        n, p = X.shape
        if penalty_weights is None:
            penalty_weights = np.ones(p)

        # 1. Calcul de lambda_max (heuristique sur la corrélation max avec y)
        # On centre y pour l'estimation de lambda_max
        lambda_max = 5*np.max(np.abs(X.T @ y))
        # max_corr = np.max(np.abs(X.T @ y))
        # lambda_max = max_corr / n # pas besoin de normalisation car le problème résolu n'est pas normalisé par n

        # Grille logarithmique décroissante
        self.lambdas_ = np.logspace(np.log10(lambda_max), np.log10(lambda_max * self.eps), self.n_lambdas)

        self.coefs_path_ = np.zeros((self.n_lambdas, p))
        self.bics_ = np.zeros(self.n_lambdas)
        intercepts = np.zeros(self.n_lambdas)

        # Pré-calculs ADMM (Cholesky)
        ATA_plus_I = X.T @ X + np.eye(p)
        L = np.linalg.cholesky(ATA_plus_I)

        # Variables d'état pour le Warm-Start (initialisées à 0 une seule fois)
        x_ws = np.zeros(p)
        z1_ws = np.zeros(n)
        z2_ws = np.zeros(p)
        u1_ws = np.zeros(n)
        u2_ws = np.zeros(p)

        def huber_prox_scaled(v):
            threshold = self.delta * (1 + 2.0 / self.rho)
            t_quad = v / (1 + 2.0 / self.rho)
            t_lin = v - (2.0 * self.delta / self.rho) * np.sign(v)
            return np.where(np.abs(v) <= threshold, t_quad, t_lin)

        # 2. Descente de la grille avec Warm-Start
        for i, lbd in enumerate(self.lambdas_):
            kappa = (lbd * penalty_weights) / self.rho

            # Boucle ADMM allégée (convergence très rapide grâce au Warm-Start)
            for k_iter in range(200):  # Max itérations réduit
                v1 = z1_ws + y - u1_ws
                v2 = z2_ws - u2_ws
                rhs = X.T @ v1 + v2
                x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

                w1 = X @ x_new - y + u1_ws
                z1_new = huber_prox_scaled(w1)

                w2 = x_new + u2_ws
                z2_new = np.sign(w2) * np.maximum(np.abs(w2) - kappa, 0)

                u1_ws = u1_ws + X @ x_new - z1_new - y
                u2_ws = u2_ws + x_new - z2_new

                # Critère d'arrêt simplifié pour le path
                if np.linalg.norm(x_new - x_ws) < 1e-4:
                    x_ws = x_new
                    break

                x_ws, z1_ws, z2_ws = x_new, z1_new, z2_new

            # Sauvegarde des résultats
            self.coefs_path_[i, :] = x_ws

            # Récupération de l'intercept (Si X est standardisé, l'intercept est la médiane des résidus)
            res_partial = y - X @ x_ws
            current_intercept = np.median(res_partial)
            intercepts[i] = current_intercept

            # Calcul du BIC
            bic_val, _ = compute_lambert_robust_bic(X, y, x_ws, current_intercept, self.delta)
            self.bics_[i] = bic_val

        # 3. Sélection du meilleur modèle
        best_idx = np.argmin(self.bics_)
        self.best_lambda_ = self.lambdas_[best_idx]
        self.best_coef_ = self.coefs_path_[best_idx, :]
        self.best_intercept_ = intercepts[best_idx]

        print("Avant SelectLambda1SE\n")
        print(f"Best lambda: {self.best_lambda_}\n")
        print(f"Best coef: {self.best_coef_}\n")
        print(f"Best intercept: {self.best_intercept_}\n")

        self.select_lambda_1se()
        print("Après SelectLambda1SE\n")
        print(f"Best lambda: {self.best_lambda_}\n")
        print(f"Best coef: {self.best_coef_}\n")
        print(f"Best intercept: {self.best_intercept_}\n")

    def plot_path(self, feature_names=None):
        """Affiche le chemin de régularisation et le minimum du BIC."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Graphe 1 : Chemin des coefficients
        l1_norms = np.sum(np.abs(self.coefs_path_), axis=1)
        for j in range(self.coefs_path_.shape[1]):
            label = feature_names[j] if feature_names else f"Facteur {j}"
            ax1.plot(self.lambdas_, self.coefs_path_[:, j], label=label, lw=2)

        ax1.axvline(self.best_lambda_, color='r', linestyle='--', label=fr'Optimal $\lambda$')
        ax1.set_xscale('log')
        ax1.invert_xaxis()  # La convention est d'afficher lambda décroissant
        ax1.set_ylabel(r'Coefficients $\beta$', fontsize=12)
        ax1.set_title('Chemin de Régularisation Warm-Start (ADMM Huber-Lasso)', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, ls='--', alpha=0.6)

        # Graphe 2 : Évolution du BIC
        ax2.plot(self.lambdas_, self.bics_, color='purple', lw=2, marker='o', markersize=4)
        ax2.axvline(self.best_lambda_, color='r', linestyle='--')
        ax2.set_xlabel(r'Paramètre de pénalité $\lambda$ (log scale)', fontsize=12)
        ax2.set_ylabel('Critère BIC Robuste', fontsize=12)
        ax2.grid(True, ls='--', alpha=0.6)

        plt.tight_layout()
        plt.draw()

    def select_lambda_1se(self):
        """
        Applique la règle de l'erreur standard (1-SE) pour sélectionner
        un lambda plus parcimonieux que le simple minimum.
        """
        # 1. Trouver le minimum du BIC
        min_idx = np.argmin(self.bics_)
        min_bic = self.bics_[min_idx]

        # 2. Estimer l'erreur standard (SE) du BIC sur le chemin
        # On utilise l'écart-type des valeurs de BIC comme approximation de l'incertitude
        se_bic = np.std(self.bics_) / np.sqrt(self.n_lambdas)

        # 3. Trouver le plus grand lambda (plus petite valeur d'index)
        # dont le BIC est <= min_bic + SE
        threshold = min_bic + se_bic

        # On cherche parmi tous les lambdas plus grands que le minimum (index < min_idx)
        # ceux qui respectent la condition
        candidates = np.where(self.bics_[:min_idx] <= threshold)[0]

        if len(candidates) > 0:
            best_idx = candidates[0]  # Le plus grand lambda qui respecte la condition
        else:
            best_idx = min_idx  # Fallback sur le minimum si aucun autre n'est trouvé

        self.best_lambda_ = self.lambdas_[best_idx]
        self.best_coef_ = self.coefs_path_[best_idx, :]

        print(f"[INFO] Sélection 1-SE appliquée. Lambda final : {self.best_lambda_:.6f}")


def estimate_universal_threshold_lambda(X, y, delta=1.35):
    """
    Calcule le paramètre lambda optimal via l'approche de Seuillage Universel
    de Donoho et Johnstone (1994), adapté pour la régression robuste.
    """
    n, p = X.shape

    # 1. Ajustement d'un modèle Huber NON PÉNALISÉ (lambda = 0)
    # Puisque p << n, ce modèle est consistant et ne sur-apprend pas.
    from sklearn.linear_model import HuberRegressor
    unpenalized_model = HuberRegressor(epsilon=delta, fit_intercept=True)
    unpenalized_model.fit(X, y)

    # 2. Extraction des résidus
    res = y - unpenalized_model.predict(X)

    # 3. Estimation robuste de l'écart-type du bruit (sigma_hat)
    # L'estimateur MAD est crucial ici car y_excess a des queues épaisses (Option A)
    mad = np.median(np.abs(res - np.median(res)))
    sigma_hat = 1.4826 * mad

    # 4. Formule de Donoho-Johnstone
    # Lambda = sigma_hat * sqrt(2 * log(p))
    # Note : On divise par n car dans notre fonction objectif ADMM/CVXPY,
    # la perte est une somme (et non une moyenne). Il faut donc ajuster l'échelle.
    lambda_universal = (sigma_hat * np.sqrt(2 * np.log(p))) / n

    print(f"[Donoho-Johnstone] Sigma_hat estimé : {sigma_hat:.4f}")
    print(f"[Donoho-Johnstone] Lambda Universel optimal calculé : {lambda_universal:.6f}")

    return lambda_universal

