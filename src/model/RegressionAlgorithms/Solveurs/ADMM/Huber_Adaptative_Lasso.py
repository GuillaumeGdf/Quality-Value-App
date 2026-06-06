import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


# ================================================================
# Fonctions ADMM (Huber LASSO)
# ================================================================
def huber_prox(v, rho, delta):
    """Proximal de la perte Huber (définition CVXPY)."""
    threshold = delta * (1 + rho)
    return np.where(np.abs(v) <= threshold,
                    v / (1 + rho),
                    v - rho * delta * np.sign(v))


def soft_threshold(v, kappa):
    """Seuillage doux."""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)


def huber_lasso_admm(A, b, lambda_, delta, rho=1.0, max_iter=2000, tol=1e-6, verbose=False):
    """Huber LASSO via ADMM (2 variables auxiliaires)."""
    m, n = A.shape
    ATA_plus_I = A.T @ A + np.eye(n)
    L = np.linalg.cholesky(ATA_plus_I)

    x = np.zeros(n)
    z1 = np.zeros(m)
    z2 = np.zeros(n)
    u1 = np.zeros(m)
    u2 = np.zeros(n)

    history = {'primal_residual': [], 'dual_residual': [], 'objective': []}

    for k in range(max_iter):
        # x-update
        v1 = z1 + b - u1
        v2 = z2 - u2
        rhs = A.T @ v1 + v2
        x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

        # z1-update (Huber proximal)
        w1 = A @ x_new - b + u1
        z1_new = huber_prox(w1, rho, delta)

        # z2-update (soft threshold)
        w2 = x_new + u2
        z2_new = soft_threshold(w2, lambda_ / rho)

        # dual update
        u1_new = u1 + A @ x_new - z1_new - b
        u2_new = u2 + x_new - z2_new

        primal_res = np.linalg.norm(A @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
        dual_res = rho * np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2))

        x, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

        resid = A @ x - b
        huber_loss = np.sum(np.where(np.abs(resid) <= delta,
                                     resid ** 2,
                                     2 * delta * np.abs(resid) - delta ** 2))
        obj = huber_loss + lambda_ * np.sum(np.abs(x))
        history['primal_residual'].append(primal_res)
        history['dual_residual'].append(dual_res)
        history['objective'].append(obj)

        if primal_res < tol and dual_res < tol:
            break

    return x, history


def huber_elastic_net_admm(A, b, lambda1, lambda2, delta, rho=1.0, max_iter=2000, tol=1e-6, verbose=False):
    """Huber Elastic Net via ADMM."""
    m, n = A.shape
    ridge_factor = 1 + 2 * lambda2 / rho
    ATA_ridge = A.T @ A + ridge_factor * np.eye(n)
    L = np.linalg.cholesky(ATA_ridge)

    x = np.zeros(n)
    z1 = np.zeros(m)
    z2 = np.zeros(n)
    u1 = np.zeros(m)
    u2 = np.zeros(n)

    history = {'primal_residual': [], 'dual_residual': [], 'objective': []}

    for k in range(max_iter):
        v1 = z1 + b - u1
        v2 = z2 - u2
        rhs = A.T @ v1 + v2
        x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

        w1 = A @ x_new - b + u1
        z1_new = huber_prox(w1, rho, delta)

        w2 = x_new + u2
        z2_new = soft_threshold(w2, lambda1 / rho)

        u1_new = u1 + A @ x_new - z1_new - b
        u2_new = u2 + x_new - z2_new

        primal_res = np.linalg.norm(A @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
        dual_res = rho * np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2))

        x, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

        resid = A @ x - b
        huber_loss = np.sum(np.where(np.abs(resid) <= delta,
                                     resid ** 2,
                                     2 * delta * np.abs(resid) - delta ** 2))
        obj = huber_loss + lambda1 * np.sum(np.abs(x)) + lambda2 * np.sum(x ** 2)
        history['primal_residual'].append(primal_res)
        history['dual_residual'].append(dual_res)
        history['objective'].append(obj)

        if primal_res < tol and dual_res < tol:
            break

    return x, history


# ================================================================
# Générateur de données réalistes « analyse fondamentale »
# ================================================================
def generate_fundamental_data(m=500, n=15, k_true=6, seed=42):
    """
    Simule des données d'analyse fondamentale :
    - Facteurs Z-scorés robustes (Student-t léger)
    - Corrélation maximale ~0.40 entre facteurs
    - Rendements Z-scorés robustes
    - Quelques outliers modérés (erreurs de reporting, événements extrêmes)
    - Distribution des coefficients : quelques grands, quelques petits
    """
    np.random.seed(seed)

    # 1. Matrice de corrélation : structure en blocs + bruit
    #    Simule des familles de ratios (rentabilité, valorisation, croissance...)
    n_blocks = 4
    block_size = n // n_blocks
    corr_matrix = np.eye(n)

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n
        for i in range(start, end):
            for j in range(i + 1, end):
                # Corrélation intra-bloc modérée (0.20 à 0.40)
                corr = 0.20 + 0.20 * np.random.random()
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    # Corrélations inter-blocs faibles (0.05 à 0.15)
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] == 0:
                corr = 0.05 + 0.10 * np.random.random()
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    max_corr = np.max(np.abs(corr_matrix - np.eye(n)))

    # 2. Génération des facteurs (distribution à queues un peu plus épaisses que normale)
    L = np.linalg.cholesky(corr_matrix + 1e-6 * np.eye(n))
    A = np.random.standard_t(df=10, size=(m, n)) @ L.T  # Student-t à 10 ddl

    # Z-score robuste (MAD scaling)
    median_A = np.median(A, axis=0)
    mad_A = np.median(np.abs(A - median_A), axis=0) * 1.4826
    A = (A - median_A) / mad_A

    # 3. Vrais coefficients (structure réaliste)
    x_true = np.zeros(n)

    # Quelques facteurs à fort impact
    strong_idx = [0, 2, 5]  # ex: P/E, ROE, Debt/Equity
    x_true[strong_idx] = [0.8, -0.6, 0.7]

    # Quelques facteurs à impact modéré
    medium_idx = [8, 11, 14]  # ex: Current Ratio, Gross Margin, FCF Yield
    x_true[medium_idx] = [0.3, -0.25, 0.35]

    true_idx = np.where(np.abs(x_true) > 1e-10)[0]

    # 4. Bruit avec outliers modérés (erreurs de données, événements extrêmes)
    noise_base = 0.4 * np.random.standard_t(df=8, size=m)  # rendements normalisés
    outlier_mask = np.random.rand(m) < 0.08  # 8% d'outliers
    noise_base[outlier_mask] += np.random.choice([-1, 1], size=np.sum(outlier_mask)) * \
                                np.random.uniform(2, 5, size=np.sum(outlier_mask))

    b = A @ x_true + noise_base

    # Z-score robuste de la cible
    b_median = np.median(b)
    b_mad = np.median(np.abs(b - b_median)) * 1.4826
    b = (b - b_median) / b_mad

    # Re-ajuster A pour que l'échelle reste cohérente après transformation de b
    # (optionnel, gardé simple ici)

    return A, b, x_true, true_idx, corr_matrix


def main_LASSO_vs_ELASTIC_NET():
    # ================================================================
    # Test principal
    # ================================================================
    print("=" * 70)
    print("HUBER LASSO vs ELASTIC NET — Contexte Analyse Fondamentale")
    print("=" * 70)

    # Paramètres
    m, n, k_true = 500, 15, 6
    delta_huber = 1.0  # Plus petit car données Z-scorées robustes → moins d'outliers extrêmes

    print(f"\nConfiguration :")
    print(f"  Observations : {m} (ex: 500 mois/trimestres)")
    print(f"  Facteurs     : {n} (ratios fondamentaux)")
    print(f"  Vrais signaux: {k_true}")
    print(f"  δ Huber      : {delta_huber} (adapté aux données robustes)")

    A, b, x_true, true_idx, corr_matrix = generate_fundamental_data(m=m, n=n, k_true=k_true)

    # Analyse du conditionnement
    corr_off_diag = np.abs(corr_matrix - np.eye(n))
    max_corr = np.max(corr_off_diag)
    mean_corr = np.mean(corr_off_diag)
    condition_number = np.linalg.cond(A.T @ A)

    print(f"\nDiagnostic de la matrice des facteurs :")
    print(f"  Corrélation maximale (hors diag.) : {max_corr:.3f}")
    print(f"  Corrélation moyenne (hors diag.)  : {mean_corr:.3f}")
    print(f"  Conditionnement de A^T A           : {condition_number:.1f}")
    print(f"  → {'BON' if condition_number < 100 else 'MOYEN' if condition_number < 1000 else 'MAUVAIS'} conditionnement")

    # Liste des lambdas à tester
    lambda_range = np.logspace(-2, 0.5, 15)  # 0.01 à ~3.16

    results_lasso = []
    results_enet = []

    print(f"\nTest de {len(lambda_range)} valeurs de λ...")
    print(f"{'λ':<10} {'LASSO err':<12} {'LASSO FP':<10} {'ENet err':<12} {'ENet FP':<10} {'Meilleur':<10}")
    print("-" * 65)

    best_lasso_err = np.inf
    best_enet_err = np.inf
    best_lasso_lambda = None
    best_enet_lambda = None
    best_lasso_x = None
    best_enet_x = None

    for lam in lambda_range:
        # Huber LASSO
        x_l, _ = huber_lasso_admm(A, b, lam, delta_huber, max_iter=2000, tol=1e-6)
        err_l = np.linalg.norm(x_l - x_true)
        fp_l = np.sum((np.abs(x_l) > 0.02) & (np.abs(x_true) < 1e-10))

        # Huber Elastic Net (λ₂ = 0.1 * λ₁, pénalité L2 légère)
        lam1, lam2 = lam, 0.1 * lam
        x_e, _ = huber_elastic_net_admm(A, b, lam1, lam2, delta_huber, max_iter=2000, tol=1e-6)
        err_e = np.linalg.norm(x_e - x_true)
        fp_e = np.sum((np.abs(x_e) > 0.02) & (np.abs(x_true) < 1e-10))

        results_lasso.append({'lambda': lam, 'error': err_l, 'fp': fp_l, 'x': x_l})
        results_enet.append({'lambda': lam, 'error': err_e, 'fp': fp_e, 'x': x_e})

        if err_l < best_lasso_err:
            best_lasso_err = err_l
            best_lasso_lambda = lam
            best_lasso_x = x_l

        if err_e < best_enet_err:
            best_enet_err = err_e
            best_enet_lambda = lam
            best_enet_x = x_e

        meilleur = "LASSO" if err_l < err_e else "ENet" if err_e < err_l else "Égal"
        print(f"{lam:<10.4f} {err_l:<12.4f} {fp_l:<10} {err_e:<12.4f} {fp_e:<10} {meilleur:<10}")

    print(
        f"\nMeilleur LASSO : λ={best_lasso_lambda:.4f}, erreur={best_lasso_err:.4f}, FP={results_lasso[np.argmin([r['error'] for r in results_lasso])]['fp']}")
    print(
        f"Meilleur ENet  : λ={best_enet_lambda:.4f}, erreur={best_enet_err:.4f}, FP={results_enet[np.argmin([r['error'] for r in results_enet])]['fp']}")

    # ================================================================
    # Graphiques
    # ================================================================
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Chemin de régularisation (erreur vs λ)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogx([r['lambda'] for r in results_lasso], [r['error'] for r in results_lasso],
                 'b-o', markersize=4, label='Huber LASSO', alpha=0.8)
    ax1.semilogx([r['lambda'] for r in results_enet], [r['error'] for r in results_enet],
                 'r-s', markersize=4, label='Huber ElasticNet', alpha=0.8)
    ax1.axvline(best_lasso_lambda, color='blue', linestyle='--', alpha=0.4)
    ax1.axvline(best_enet_lambda, color='red', linestyle='--', alpha=0.4)
    ax1.set_xlabel('λ')
    ax1.set_ylabel('‖x - x_true‖₂')
    ax1.set_title('Erreur de reconstruction vs λ')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Faux positifs vs λ
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogx([r['lambda'] for r in results_lasso], [r['fp'] for r in results_lasso],
                 'b-o', markersize=4, label='Huber LASSO', alpha=0.8)
    ax2.semilogx([r['lambda'] for r in results_enet], [r['fp'] for r in results_enet],
                 'r-s', markersize=4, label='Huber ElasticNet', alpha=0.8)
    ax2.set_xlabel('λ')
    ax2.set_ylabel('Faux positifs')
    ax2.set_title('Sélection de variables (FP)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Coefficients optimaux (LASSO)
    ax3 = fig.add_subplot(gs[0, 2])
    x_idx = np.arange(n)
    width = 0.35
    ax3.bar(x_idx - width / 2, x_true, width, color='green', alpha=0.9, label='Vrai')
    ax3.bar(x_idx + width / 2, best_lasso_x, width, color='blue', alpha=0.7, label=f'LASSO (λ={best_lasso_lambda:.3f})')
    ax3.set_xlabel('Facteur')
    ax3.set_ylabel('Coefficient')
    ax3.set_title(f'Huber LASSO optimal\nErreur = {best_lasso_err:.4f}')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Coefficients optimaux (ENet)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.bar(x_idx - width / 2, x_true, width, color='green', alpha=0.9, label='Vrai')
    ax4.bar(x_idx + width / 2, best_enet_x, width, color='red', alpha=0.7, label=f'ENet (λ₁={best_enet_lambda:.3f})')
    ax4.set_xlabel('Facteur')
    ax4.set_ylabel('Coefficient')
    ax4.set_title(f'Huber Elastic Net optimal\nErreur = {best_enet_err:.4f}')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Comparaison des erreurs minimales
    ax5 = fig.add_subplot(gs[1, 0])
    methods = ['Huber LASSO', 'Huber Elastic Net']
    errors = [best_lasso_err, best_enet_err]
    colors = ['blue', 'red']
    bars = ax5.bar(methods, errors, color=colors, alpha=0.7, width=0.4)
    for bar, err in zip(bars, errors):
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                 f'{err:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax5.set_ylabel('‖x - x_true‖₂ minimale')
    ax5.set_title('Meilleure erreur atteignable')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Corrélogramme des facteurs
    ax6 = fig.add_subplot(gs[1, 1])
    im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='equal')
    ax6.set_title('Matrice de corrélation des facteurs\n(max hors diag = {:.2f})'.format(max_corr))
    plt.colorbar(im, ax=ax6, shrink=0.8)

    # 7. Distribution des coefficients vrais
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.stem(x_idx, x_true, linefmt='g-', markerfmt='go', basefmt=' ')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax7.set_xlabel('Facteur')
    ax7.set_ylabel('Coefficient vrai')
    ax7.set_title(f'Vrais coefficients ({k_true} non nuls)')
    ax7.grid(True, alpha=0.3)

    # 8. Résumé textuel
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    summary_text = (
        f"RÉSUMÉ — Contexte Analyse Fondamentale\n"
        f"{'─' * 35}\n"
        f"Observations (m)      : {m}\n"
        f"Facteurs (n)          : {n}\n"
        f"Signaux réels         : {k_true}\n"
        f"Corrélation max       : {max_corr:.3f}\n"
        f"Conditionnement       : {condition_number:.0f}\n"
        f"{'─' * 35}\n"
        f"Meilleur LASSO\n"
        f"  λ = {best_lasso_lambda:.4f}\n"
        f"  Erreur = {best_lasso_err:.4f}\n"
        f"  FP = {results_lasso[np.argmin([r['error'] for r in results_lasso])]['fp']}\n"
        f"{'─' * 35}\n"
        f"Meilleur Elastic Net\n"
        f"  λ₁ = {best_enet_lambda:.4f}\n"
        f"  Erreur = {best_enet_err:.4f}\n"
        f"  FP = {results_enet[np.argmin([r['error'] for r in results_enet])]['fp']}\n"
        f"{'─' * 35}\n"
        f"CONCLUSION : {'LASSO suffisant' if best_lasso_err <= best_enet_err * 1.02 else 'ENet bénéfique'}\n"
        f"Écart erreur : {abs(best_lasso_err - best_enet_err):.4f}"
    )
    ax8.text(0, 1, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Huber LASSO vs Elastic Net — Contexte Réaliste d\'Analyse Fondamentale\n'
                 'Facteurs Z-scorés robustes, corrélation maîtrisée ≤ 0.40',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.savefig('fundamental_analysis_lasso_vs_enet.png', dpi=150, bbox_inches='tight')
    plt.show()


import numpy as np
from scipy.optimize import minimize


def compute_lambda_max_huber(A, b, delta):
    """
    Calcule le plus petit λ tel que la solution du Huber LASSO soit 0.
    Condition : ||A^T ψ(-b)||_∞ ≤ λ  où ψ est la dérivée de Huber.
    """

    def huber_gradient(r):
        """Dérivée de la perte Huber (élément par élément)."""
        return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    # Au point x=0, le résidu est -b
    grad_at_zero = A.T @ huber_gradient(-b)
    return np.max(np.abs(grad_at_zero))


def dpp_screening_huber(A, b, lam, delta, previous_x=None, rho=1.0):
    """
    Screening rule DPP pour le Huber LASSO.

    Retourne l'ensemble des indices potentiellement actifs.
    Les variables exclues sont garanties nulles à l'optimum.

    Parameters :
    - lam : valeur courante de λ
    - previous_x : solution au λ précédent (pour warm start et bornes)
    - rho : paramètre pour la borne duale
    """
    m, n = A.shape

    # Si pas de solution précédente, on garde tout
    if previous_x is None:
        return np.arange(n)

    # Calcul du résidu et du gradient dual au point précédent
    resid = A @ previous_x - b
    psi = np.where(np.abs(resid) <= delta, resid, delta * np.sign(resid))
    theta_prev = psi  # Variable duale (scaled)

    # Calcul du rayon de la boule duale
    # Basé sur la décroissance de la fonction objectif
    primal_obj = np.sum(np.where(np.abs(resid) <= delta,
                                 0.5 * resid ** 2,
                                 delta * (np.abs(resid) - 0.5 * delta)))
    dual_gap = primal_obj + lam * np.sum(np.abs(previous_x))  # Gap d'optimalité

    # Rayon de la boule contenant la solution duale optimale
    radius = np.sqrt(2 * dual_gap / rho)

    # Screening : pour chaque variable j, on calcule la borne supérieure de |A_j^T θ*|
    # Si la borne < lam, la variable est inactive
    safe_set = []

    for j in range(n):
        a_j = A[:, j]
        # Centre : A_j^T θ_prev
        center = np.dot(a_j, theta_prev)
        # Rayon : ||a_j|| * radius (inégalité de Cauchy-Schwarz)
        bound = np.abs(center) + np.linalg.norm(a_j) * radius

        if bound >= lam * 0.99:  # Marge de sécurité de 1%
            safe_set.append(j)

    # Si on a trop agressivement réduit, on garde tout
    if len(safe_set) < 2:
        return np.arange(n)

    return np.array(safe_set)


def huber_lasso_path_admm(A, b, delta, lambda_range=None, n_lambda=50,
                          eps=1e-4, rho=1.0, max_iter=2000, tol=1e-6, verbose=True):
    """
    Chemin de régularisation complet pour le Huber LASSO.
    Utilise le screening DPP et le warm start pour accélérer.
    """
    m, n = A.shape

    # 1. λ_max
    lambda_max = compute_lambda_max_huber(A, b, delta)

    # 2. Séquence de λ
    if lambda_range is None:
        lambda_min = eps * lambda_max
        lambda_range = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)
    else:
        n_lambda = len(lambda_range)

    # 3. Initialisation
    x_current = np.zeros(n)  # Pour λ_max, x* = 0
    solutions = np.zeros((n_lambda, n))
    active_sizes = np.zeros(n_lambda)
    screened_sizes = np.zeros(n_lambda)

    # Pré-factorisation pour ADMM (matrice complète)
    ATA_plus_I = A.T @ A + np.eye(n)
    L_full = np.linalg.cholesky(ATA_plus_I)

    if verbose:
        print(f"λ_max = {lambda_max:.4f}")
        print(f"Chemin de {n_lambda} λ de {lambda_min:.6f} à {lambda_max:.4f}")
        print(f"{'λ':<12} {'Actifs':<10} {'Screenés':<12} {'Itérations':<12}")
        print("-" * 50)

    for i, lam in enumerate(lambda_range):
        # Screening
        if i == 0:
            safe_set = np.arange(n)  # Premier λ : tout est actif
        else:
            safe_set = dpp_screening_huber(A, b, lam, delta, x_current, rho)

        screened_sizes[i] = len(safe_set)

        # Si l'ensemble safe est assez petit, on réduit la matrice
        if len(safe_set) < n * 0.8:  # 80% de réduction minimum
            A_safe = A[:, safe_set]
            # Re-factorisation rapide
            ATA_safe = A_safe.T @ A_safe + np.eye(len(safe_set))
            L = np.linalg.cholesky(ATA_safe)

            # Résolution ADMM sur les variables safe uniquement
            x_safe = x_current[safe_set]
            z1 = np.zeros(m)
            z2 = np.zeros(len(safe_set))
            u1 = np.zeros(m)
            u2 = np.zeros(len(safe_set))

            for k in range(max_iter):
                v1 = z1 + b - u1
                v2 = z2 - u2
                rhs = A_safe.T @ v1 + v2
                x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

                w1 = A_safe @ x_new - b + u1
                z1_new = np.where(np.abs(w1) <= delta * (1 + rho),
                                  w1 / (1 + rho),
                                  w1 - rho * delta * np.sign(w1))

                w2 = x_new + u2
                z2_new = np.sign(w2) * np.maximum(np.abs(w2) - lam / rho, 0)

                u1_new = u1 + A_safe @ x_new - z1_new - b
                u2_new = u2 + x_new - z2_new

                primal_res = np.linalg.norm(A_safe @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
                dual_res = rho * np.linalg.norm(A_safe.T @ (z1_new - z1) + (z2_new - z2))

                x_safe, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

                if primal_res < tol and dual_res < tol:
                    break

            x_current = np.zeros(n)
            x_current[safe_set] = x_safe

        else:
            # Résolution complète
            x_current, hist = huber_lasso_admm(A, b, lam, delta, rho=rho,
                                               max_iter=max_iter, tol=tol)
            k = len(hist['objective'])

        solutions[i] = x_current
        active_sizes[i] = np.sum(np.abs(x_current) > 1e-6)

        if verbose:
            k_val = len(safe_set) if len(safe_set) < n * 0.8 else k
            print(
                f"{lam:<12.6f} {active_sizes[i]:<10.0f} {screened_sizes[i]:<12.0f} {k_val if isinstance(k_val, int) else len(safe_set):<12}")

    return lambda_range, solutions, active_sizes, screened_sizes


def main_LASSO_vs_DPP_RULE():
    # ================================================================
    # Test principal
    # ================================================================
    print("=" * 70)
    print("HUBER LASSO vs ELASTIC NET — Contexte Analyse Fondamentale")
    print("=" * 70)

    # Paramètres
    m, n, k_true = 500, 50, 25
    delta_huber = 1.0  # Plus petit car données Z-scorées robustes → moins d'outliers extrêmes

    print(f"\nConfiguration :")
    print(f"  Observations : {m} (ex: 500 mois/trimestres)")
    print(f"  Facteurs     : {n} (ratios fondamentaux)")
    print(f"  Vrais signaux: {k_true}")
    print(f"  δ Huber      : {delta_huber} (adapté aux données robustes)")

    A, b, x_true, true_idx, corr_matrix = generate_fundamental_data(m=m, n=n, k_true=k_true)

    # Analyse du conditionnement
    corr_off_diag = np.abs(corr_matrix - np.eye(n))
    max_corr = np.max(corr_off_diag)
    mean_corr = np.mean(corr_off_diag)
    condition_number = np.linalg.cond(A.T @ A)

    print(f"\nDiagnostic de la matrice des facteurs :")
    print(f"  Corrélation maximale (hors diag.) : {max_corr:.3f}")
    print(f"  Corrélation moyenne (hors diag.)  : {mean_corr:.3f}")
    print(f"  Conditionnement de A^T A           : {condition_number:.1f}")
    print(
        f"  → {'BON' if condition_number < 100 else 'MOYEN' if condition_number < 1000 else 'MAUVAIS'} conditionnement")

    # Liste des lambdas à tester
    lambda_range = np.logspace(-2, 0.5, 15)  # 0.01 à ~3.16

    results_lasso = []
    results_enet = []

    print(f"\nTest de {len(lambda_range)} valeurs de λ...")
    print(f"{'λ':<10} {'LASSO err':<12} {'LASSO FP':<10} {'ENet err':<12} {'ENet FP':<10} {'Meilleur':<10}")
    print("-" * 65)

    best_lasso_err = np.inf
    best_enet_err = np.inf
    best_lasso_lambda = None
    best_enet_lambda = None
    best_lasso_x = None
    best_enet_x = None

    # Calcul du chemin complet
    lambda_range, solutions, active_sizes, screened_sizes = huber_lasso_path_admm(
        A, b, delta=1.0, n_lambda=50, verbose=True
    )

    # Graphique du chemin de régularisation
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Chemin des coefficients
    ax = axes[0]
    for j in range(n):
        ax.semilogx(lambda_range, solutions[:, j], alpha=0.7,
                    linewidth=2 if j in true_idx else 0.8,
                    color='blue' if j in true_idx else 'gray')
    ax.set_xlabel('λ')
    ax.set_ylabel('Coefficient')
    ax.set_title('Chemin de régularisation')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # 2. Taille de l'ensemble actif
    ax = axes[1]
    ax.semilogx(lambda_range, active_sizes, 'b-o', markersize=3)
    ax.set_xlabel('λ')
    ax.set_ylabel('Nombre de variables actives')
    ax.set_title('Parcimonie le long du chemin')
    ax.grid(True, alpha=0.3)

    # 3. Efficacité du screening
    ax = axes[2]
    ax.fill_between(lambda_range, screened_sizes, alpha=0.3, label='Screenées')
    ax.fill_between(lambda_range, active_sizes, screened_sizes, alpha=0.3, label='Éliminées')
    ax.semilogx(lambda_range, active_sizes, 'r-', label='Actives')
    ax.set_xlabel('λ')
    ax.set_ylabel('Nombre de variables')
    ax.set_title('Efficacité du screening DPP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('huber_lasso_path.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    # main_LASSO_vs_ELASTIC_NET()
    main_LASSO_vs_DPP_RULE()