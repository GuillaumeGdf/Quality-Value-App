import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from scipy.sparse.linalg import cg, LinearOperator
import warnings

warnings.filterwarnings('ignore')


# ================================================================
# Fonctions ADMM
# ================================================================
def huber_prox(v, rho, delta):
    threshold = delta * (1 + rho)
    return np.where(np.abs(v) <= threshold, v / (1 + rho), v - rho * delta * np.sign(v))


def soft_threshold(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)


def huber_lasso_admm_highdim(A, b, lambda_, delta, rho=1.0, max_iter=2000, tol=1e-5,
                             x_init=None, verbose=False):
    m, n = A.shape
    x = np.zeros(n) if x_init is None else x_init.copy()
    z1, z2 = np.zeros(m), np.zeros(n)
    u1, u2 = np.zeros(m), np.zeros(n)

    for k in range(max_iter):
        v1, v2 = z1 + b - u1, z2 - u2
        rhs = A.T @ v1 + v2
        A_op = LinearOperator((n, n), matvec=lambda xv: A.T @ (A @ xv) + xv, dtype=A.dtype)
        x_new, info = cg(A_op, rhs, x0=x, rtol=1e-8, maxiter=min(150, n))
        if info != 0:
            x_new = x - 0.5 * (A.T @ (A @ x - v1) + (x - v2))

        w1 = A @ x_new - b + u1
        z1_new = huber_prox(w1, rho, delta)
        w2 = x_new + u2
        z2_new = soft_threshold(w2, lambda_ / rho)
        u1_new = u1 + A @ x_new - z1_new - b
        u2_new = u2 + x_new - z2_new

        primal_res = np.linalg.norm(A @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
        dual_res = rho * np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2))
        x, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

        if primal_res < tol and dual_res < tol:
            break
    return x, {'iterations': k + 1}


# ================================================================
# Screening DPP
# ================================================================
def compute_lambda_max_huber(A, b, delta):
    def huber_gradient(r):
        return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    return np.max(np.abs(A.T @ huber_gradient(-b)))


def dpp_screening_huber_fast(A, b, lam, delta, previous_x, rho=1.0, safety_margin=0.99):
    m, n = A.shape
    if previous_x is None:
        return np.arange(n), n

    resid = A @ previous_x - b
    psi = np.where(np.abs(resid) <= delta, resid, delta * np.sign(resid))
    primal_obj = np.sum(np.where(np.abs(resid) <= delta, 0.5 * resid ** 2, delta * (np.abs(resid) - 0.5 * delta)))
    dual_gap = primal_obj + lam * np.sum(np.abs(previous_x))
    radius = np.sqrt(2 * dual_gap / max(rho, 1e-8))

    centers = A.T @ psi
    norms_A = np.linalg.norm(A, axis=0)
    bounds = np.abs(centers) + norms_A * radius
    safe_mask = bounds >= lam * safety_margin
    safe_set = np.where(safe_mask)[0]
    if len(safe_set) < 2:
        safe_set = np.arange(n)
    return safe_set, len(safe_set)


def huber_lasso_path_dpp(A, b, delta, lambda_range=None, n_lambda=40, eps=1e-3, rho=1.0,
                         max_iter=2000, tol=1e-5, verbose=True):
    m, n = A.shape
    lambda_max = compute_lambda_max_huber(A, b, delta)
    if lambda_range is None:
        lambda_min = eps * lambda_max
        lambda_range = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)
    else:
        n_lambda = len(lambda_range)

    solutions = np.zeros((n_lambda, n))
    active_sizes = np.zeros(n_lambda, dtype=int)
    screened_sizes = np.zeros(n_lambda, dtype=int)
    times = np.zeros(n_lambda)
    x_current = np.zeros(n)

    if verbose:
        print(f"λ_max={lambda_max:.4f}, λ_min={lambda_range[-1]:.6f}, {n_lambda} valeurs")
        print(f"{'λ':<12} {'Actifs':<8} {'Screenés':<8} {'Éliminés':<8} {'Temps':<8}")
        print(f"{'-' * 44}")

    for i, lam in enumerate(lambda_range):
        t_start = time.time()
        if i == 0:
            safe_set = np.arange(n)
            screened_sizes[i] = n
        else:
            safe_set, screened_sizes[i] = dpp_screening_huber_fast(A, b, lam, delta, x_current, rho)

        if len(safe_set) < n * 0.95:
            A_safe = A[:, safe_set]
            x_safe_init = x_current[safe_set]
            n_safe = len(safe_set)
            A_op_safe = LinearOperator((n_safe, n_safe),
                                       matvec=lambda xv: A_safe.T @ (A_safe @ xv) + xv,
                                       dtype=A.dtype)
            x_safe = x_safe_init.copy()
            z1, z2 = np.zeros(m), np.zeros(n_safe)
            u1, u2 = np.zeros(m), np.zeros(n_safe)
            for k in range(max_iter):
                v1, v2 = z1 + b - u1, z2 - u2
                rhs = A_safe.T @ v1 + v2
                x_new, info = cg(A_op_safe, rhs, x0=x_safe, rtol=1e-8, maxiter=min(150, n_safe))
                if info != 0:
                    x_new = x_safe - 0.5 * (A_safe.T @ (A_safe @ x_safe - v1) + (x_safe - v2))
                w1 = A_safe @ x_new - b + u1
                z1_new = huber_prox(w1, rho, delta)
                w2 = x_new + u2
                z2_new = soft_threshold(w2, lam / rho)
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
            x_current, _ = huber_lasso_admm_highdim(A, b, lam, delta, rho=rho,
                                                    max_iter=max_iter, tol=tol, x_init=x_current)
        solutions[i] = x_current
        active_sizes[i] = np.sum(np.abs(x_current) > 1e-5)
        times[i] = time.time() - t_start
        if verbose:
            eliminated = n - screened_sizes[i]
            print(f"{lam:<12.6f} {active_sizes[i]:<8} {screened_sizes[i]:<8} {eliminated:<8} {times[i]:<8.4f}")

    if verbose:
        print(f"{'=' * 44}")
        print(f"Temps total : {np.sum(times):.2f}s")
        eliminated_avg = np.mean(n - screened_sizes)
        print(f"Variables éliminées moy. : {eliminated_avg:.0f}/{n} ({100 * eliminated_avg / n:.1f}%)")
    return lambda_range, solutions, active_sizes, screened_sizes, times


# ================================================================
# Générateur BIEN CONDITIONNÉ
# ================================================================
def generate_well_conditioned_array(m=32, n_directions=None, k_sources=25, snr_db=15, seed=42):
    """
    Antenne ULA bien conditionnée.
    - m capteurs, d = λ/2
    - n_directions = 2m à 3m (grille de Nyquist)
    - Sources bien espacées
    """
    np.random.seed(seed)

    d = 0.5  # λ/2

    # Grille adaptée au nombre de capteurs (Nyquist)
    if n_directions is None:
        n_directions = 3 * m  # 96 directions pour 32 capteurs

    theta_grid = np.linspace(-90, 90, n_directions)
    theta_rad = np.deg2rad(theta_grid)

    # Matrice de steering vectors
    sensor_idx = np.arange(m)
    A_complex = np.exp(-1j * 2 * np.pi * d * np.outer(sensor_idx, np.sin(theta_rad)))
    A_complex = A_complex / np.sqrt(m)

    # Espacement minimal entre sources
    min_sep_deg = 180 / n_directions * 3  # au moins 3 bins d'écart
    min_sep_idx = max(2, int(min_sep_deg * n_directions / 180))

    available = list(range(n_directions))
    source_indices = []
    for _ in range(k_sources):
        if not available:
            break
        idx = np.random.choice(available)
        source_indices.append(idx)
        to_remove = [i for i in available if abs(i - idx) < min_sep_idx]
        available = [i for i in available if i not in to_remove]

    source_indices = np.array(sorted(source_indices))
    k_actual = len(source_indices)

    # Amplitudes (puissance décroissante)
    amplitudes = np.zeros(n_directions, dtype=complex)
    powers = np.sort(np.random.exponential(1.0, size=k_actual))[::-1]
    for i, idx in enumerate(source_indices):
        phase = np.random.uniform(0, 2 * np.pi)
        amplitudes[idx] = np.sqrt(powers[i]) * np.exp(1j * phase)

    # Signal reçu
    y_clean = A_complex @ amplitudes

    # Bruit
    noise_power = np.mean(np.abs(y_clean) ** 2) / (10 ** (snr_db / 10))
    noise_gauss = np.sqrt(noise_power / 2) * (np.random.randn(m) + 1j * np.random.randn(m))

    # Impulsions Cauchy
    impulse_prob = 0.05
    impulse_mask = np.random.rand(m) < impulse_prob
    noise_impulse = np.zeros(m, dtype=complex)
    n_imp = np.sum(impulse_mask)
    if n_imp > 0:
        noise_impulse[impulse_mask] = (np.random.standard_cauchy(n_imp) +
                                       1j * np.random.standard_cauchy(n_imp))
        noise_impulse[impulse_mask] *= 5 * np.sqrt(noise_power)

    y = y_clean + noise_gauss + noise_impulse

    # Conversion en réel
    A_real = np.vstack([np.real(A_complex), np.imag(A_complex)])
    y_real = np.concatenate([np.real(y), np.imag(y)])
    x_real = np.real(amplitudes)

    # Normalisation robuste
    y_median = np.median(y_real)
    y_mad = np.median(np.abs(y_real - y_median)) * 1.4826
    y_real = (y_real - y_median) / max(y_mad, 1e-8)

    for j in range(n_directions):
        col = A_real[:, j]
        col_median = np.median(col)
        col_mad = np.median(np.abs(col - col_median)) * 1.4826
        if col_mad > 1e-8:
            A_real[:, j] = (col - col_median) / col_mad

    # Analyse de cohérence
    A_norm = A_real / np.maximum(np.linalg.norm(A_real, axis=0), 1e-10)
    G = np.abs(A_norm.T @ A_norm)
    np.fill_diagonal(G, 0)
    mutual_coherence = np.max(G)
    avg_coherence = np.mean(G)

    print(f"\nDonnées générées (Array Processing bien conditionné) :")
    print(f"  Capteurs (m)        : {m}")
    print(f"  Directions (n)      : {n_directions}  (n/m = {n_directions / m:.1f})")
    print(f"  Espacement          : d = {d}λ")
    print(f"  Sources actives     : {k_actual}")
    print(f"  SNR                 : {snr_db} dB")
    print(f"  Impulsions          : {100 * np.mean(impulse_mask):.1f}%")
    print(f"  Cohérence mutuelle  : {mutual_coherence:.4f}")
    print(f"  Cohérence moyenne   : {avg_coherence:.4f}")

    if mutual_coherence < 0.5:
        print(f"  → BONNE : LASSO efficace, DPP puissant")
    elif mutual_coherence < 0.9:
        print(f"  → MOYENNE : LASSO possible, DPP modéré")
    else:
        print(f"  → MAUVAISE : Grille trop fine, réduire n/m")

    return A_real, y_real, x_real, source_indices, theta_grid, G


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HUBER LASSO + DPP — Détection de sources (Array Processing)")
    print("Antenne ULA : d=λ/2, 32 capteurs, 25 sources")
    print("Grille adaptée : n = 3m = 96 directions")
    print("=" * 70)

    M = 32
    K_SOURCES = 25
    SNR_DB = 15
    N_DIR = 3 * M  # 96 directions (grille de Nyquist)

    A, b, x_true, source_indices, theta_grid, G = generate_well_conditioned_array(
        m=M, n_directions=N_DIR, k_sources=K_SOURCES, snr_db=SNR_DB, seed=42
    )

    m_real, n_grid = A.shape
    k_actual = len(source_indices)
    delta_huber = 1.0

    # Cohérence hors-diagonale
    G_off = G.copy()
    np.fill_diagonal(G_off, 0)
    mutual_coh = np.max(G_off)

    # ================================================================
    # Chemin DPP
    # ================================================================
    print("\n[PARTIE 1] Chemin de régularisation AVEC screening DPP")
    lambda_range, solutions_dpp, active_dpp, screened_dpp, times_dpp = \
        huber_lasso_path_dpp(A, b, delta_huber, n_lambda=40, eps=1e-3, verbose=True)

    # ================================================================
    # Comparaison DPP vs Full
    # ================================================================
    print("\n[PARTIE 2] Comparaison DPP vs Full")
    test_indices = np.linspace(0, len(lambda_range) - 1, 6, dtype=int)
    test_lambdas = lambda_range[test_indices]

    comp = []
    for lam in test_lambdas:
        idx = np.argmin(np.abs(lambda_range - lam))
        x_dpp = solutions_dpp[idx]
        t_dpp = times_dpp[idx]

        print(f"  λ={lam:.4f} (full)...", end=" ", flush=True)
        x_full, _ = huber_lasso_admm_highdim(A, b, lam, delta_huber, max_iter=2000, tol=1e-5)
        diff = np.linalg.norm(x_dpp - x_full)
        comp.append({'lambda': lam, 'diff': diff, 't_dpp': t_dpp})
        print(f"diff={diff:.2e}")

    # ================================================================
    # Détection
    # ================================================================
    print("\n[PARTIE 3] Détection de sources")
    best_f1 = 0
    best_idx = 0
    best_threshold = 1e-3

    for i in range(len(lambda_range)):
        x_est = solutions_dpp[i]
        mx = np.max(np.abs(x_est))
        if mx < 1e-8:
            continue
        threshold = 0.1 * mx
        detected = np.where(np.abs(x_est) > threshold)[0]
        tp = len(set(detected) & set(source_indices))
        fp = len(set(detected) - set(source_indices))
        fn = len(set(source_indices) - set(detected))
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_idx = i
                best_threshold = threshold

    x_opt = solutions_dpp[best_idx]
    threshold_opt = best_threshold
    detected_opt = np.where(np.abs(x_opt) > threshold_opt)[0]
    tp_opt = len(set(detected_opt) & set(source_indices))
    fp_opt = len(set(detected_opt) - set(source_indices))
    fn_opt = len(set(source_indices) - set(detected_opt))
    prec_opt = tp_opt / (tp_opt + fp_opt) * 100 if (tp_opt + fp_opt) > 0 else 0
    rec_opt = tp_opt / (tp_opt + fn_opt) * 100 if (tp_opt + fn_opt) > 0 else 0

    print(f"  λ optimal         : {lambda_range[best_idx]:.4f}")
    print(f"  F1-score          : {best_f1:.3f}")
    print(f"  TP={tp_opt}/{k_actual}, FP={fp_opt}, FN={fn_opt}")
    print(f"  Précision = {prec_opt:.1f}%, Rappel = {rec_opt:.1f}%")

    # ================================================================
    # GRAPHIQUES
    # ================================================================
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Spectre spatial
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(theta_grid, np.abs(x_opt), 'b-', linewidth=1, alpha=0.8, label='Spectre estimé')
    ax1.plot(theta_grid[source_indices], np.abs(x_opt)[source_indices], 'ro',
             markersize=10, markerfacecolor='none', markeredgewidth=2.5,
             label=f'Vraies sources ({k_actual})')
    ax1.plot(theta_grid[detected_opt], np.abs(x_opt)[detected_opt], 'g^',
             markersize=8, alpha=0.7, label=f'Détectées ({len(detected_opt)})')
    ax1.axhline(y=threshold_opt, color='orange', linestyle='--', alpha=0.5,
                label=f'Seuil = {threshold_opt:.2e}')
    ax1.set_xlabel('Angle d\'arrivée (degrés)')
    ax1.set_ylabel('Amplitude estimée')
    ax1.set_title(f'Pseudo-spectre spatial (λ={lambda_range[best_idx]:.4f})\n'
                  f'TP={tp_opt}, FP={fp_opt}, FN={fn_opt} | Préc={prec_opt:.0f}%, Rappel={rec_opt:.0f}%, F1={best_f1:.3f}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Matrice de cohérence
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(G, cmap='hot', aspect='equal', vmin=0, vmax=1)
    ax2.set_title(f'Cohérence G = |A^T A|\nMax hors-diag = {mutual_coh:.4f}')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. Distribution de la cohérence
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.hist(G_off.ravel(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=mutual_coh, color='red', linestyle='--', label=f'Max = {mutual_coh:.3f}')
    ax3.set_xlabel('Cohérence mutuelle')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Distribution de la cohérence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Efficacité du screening DPP
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.fill_between(lambda_range, 0, active_dpp, alpha=0.4, color='red', label='Actives')
    ax4.fill_between(lambda_range, active_dpp, screened_dpp, alpha=0.3, color='orange', label='Screenées')
    ax4.fill_between(lambda_range, screened_dpp, n_grid, alpha=0.2, color='green', label='Éliminées (DPP)')
    ax4.semilogx(lambda_range, active_dpp, 'r-', linewidth=2)
    ax4.semilogx(lambda_range, screened_dpp, 'orange', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('λ')
    ax4.set_ylabel('Nombre de directions')
    ax4.set_title('Efficacité du screening DPP')
    ax4.legend(fontsize=7, loc='upper left')
    ax4.grid(True, alpha=0.3)

    # 5. Temps de calcul
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogx(lambda_range, times_dpp, 'b-o', markersize=3, alpha=0.7)
    ax5.set_xlabel('λ')
    ax5.set_ylabel('Temps (s)')
    ax5.set_title(f'Temps par λ (total = {np.sum(times_dpp):.1f}s)')
    ax5.grid(True, alpha=0.3)

    # 6. Détection vs λ
    ax6 = fig.add_subplot(gs[1, 2])
    tp_list, fp_list, fn_list = [], [], []
    for i in range(len(lambda_range)):
        x_e = solutions_dpp[i]
        mx = np.max(np.abs(x_e))
        th = 0.1 * mx if mx > 1e-8 else 1e-3
        det = np.where(np.abs(x_e) > th)[0]
        tp_list.append(len(set(det) & set(source_indices)))
        fp_list.append(len(set(det) - set(source_indices)))
        fn_list.append(len(set(source_indices) - set(det)))
    ax6.semilogx(lambda_range, tp_list, 'g-o', markersize=3, label=f'TP (/{k_actual})')
    ax6.semilogx(lambda_range, fp_list, 'r-s', markersize=3, label='FP')
    ax6.semilogx(lambda_range, fn_list, 'orange', linestyle=':', marker='d', markersize=3, label='FN')
    ax6.axvline(lambda_range[best_idx], color='blue', linestyle='--', label=f'λ opt = {lambda_range[best_idx]:.4f}')
    ax6.set_xlabel('λ')
    ax6.set_ylabel('Nombre')
    ax6.set_title('Détection vs λ')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. Score F1 vs λ
    ax7 = fig.add_subplot(gs[1, 3])
    f1_list = []
    for i in range(len(lambda_range)):
        x_e = solutions_dpp[i]
        mx = np.max(np.abs(x_e))
        th = 0.1 * mx if mx > 1e-8 else 1e-3
        det = np.where(np.abs(x_e) > th)[0]
        tp = len(set(det) & set(source_indices))
        fp = len(set(det) - set(source_indices))
        fn = len(set(source_indices) - set(det))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_list.append(f1)
    ax7.semilogx(lambda_range, f1_list, 'b-o', markersize=3)
    ax7.axvline(lambda_range[best_idx], color='red', linestyle='--', label=f'Meilleur F1 = {best_f1:.3f}')
    ax7.set_xlabel('λ')
    ax7.set_ylabel('F1-score')
    ax7.set_title('Score F1 vs λ')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Distribution des amplitudes
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.hist(np.abs(x_opt)[source_indices], bins=15, color='green', alpha=0.6,
             edgecolor='black', label=f'Vraies sources ({k_actual})')
    non_src = list(set(range(n_grid)) - set(source_indices))
    ax8.hist(np.abs(x_opt)[non_src], bins=30, color='gray', alpha=0.4,
             edgecolor='black', label='Non-sources')
    ax8.axvline(x=threshold_opt, color='orange', linestyle='--', linewidth=2, label='Seuil')
    ax8.set_xlabel('|Amplitude estimée|')
    ax8.set_ylabel('Fréquence')
    ax8.set_title('Distribution des amplitudes')
    ax8.set_yscale('log')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)

    # 9. Matrice de confusion
    ax9 = fig.add_subplot(gs[2, 1])
    conf = np.array([[tp_opt, fp_opt], [fn_opt, n_grid - tp_opt - fp_opt - fn_opt]])
    im9 = ax9.imshow(conf, cmap='Blues', aspect='auto')
    ax9.set_xticks([0, 1])
    ax9.set_xticklabels(['Détecté', 'Non détecté'])
    ax9.set_yticks([0, 1])
    ax9.set_yticklabels(['Vraie source', 'Non-source'])
    ax9.set_title('Matrice de confusion')
    for i in range(2):
        for j in range(2):
            ax9.text(j, i, str(conf[i, j]), ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if conf[i, j] > n_grid / 4 else 'black')
    plt.colorbar(im9, ax=ax9, shrink=0.8)

    # 10. Différence DPP vs Full
    ax10 = fig.add_subplot(gs[2, 2])
    diffs = [c['diff'] for c in comp]
    ax10.semilogx([c['lambda'] for c in comp], diffs, 'r-o', markersize=5)
    ax10.set_xlabel('λ')
    ax10.set_ylabel('‖x_dpp - x_full‖₂')
    ax10.set_title('Précision du screening DPP')
    ax10.grid(True, alpha=0.3)

    # 11. Résumé
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    eliminated_avg = np.mean(n_grid - screened_dpp)
    summary = (
        f"RÉSUMÉ — Détection de sources\n"
        f"{'─' * 45}\n"
        f"Antenne : {M} capteurs ULA | d=λ/2\n"
        f"Grille : {n_grid} directions (n/m={n_grid / M:.1f})\n"
        f"Sources : {k_actual} actives | SNR={SNR_DB}dB\n"
        f"Cohérence max : {mutual_coh:.4f}\n"
        f"Cohérence moy : {np.mean(G_off):.4f}\n"
        f"{'─' * 45}\n"
        f"DPP : {eliminated_avg:.0f} dir. éliminées\n"
        f"      ({100 * eliminated_avg / n_grid:.1f}%)\n"
        f"Temps total : {np.sum(times_dpp):.1f}s\n"
        f"{'─' * 45}\n"
        f"DÉTECTION (F1 optimal)\n"
        f"λ* = {lambda_range[best_idx]:.4f}\n"
        f"TP={tp_opt}/{k_actual} | FP={fp_opt} | FN={fn_opt}\n"
        f"Précision = {prec_opt:.1f}%\n"
        f"Rappel = {rec_opt:.1f}%\n"
        f"F1 = {best_f1:.3f}"
    )
    ax11.text(0.5, 0.5, summary, transform=ax11.transAxes, fontsize=9,
              verticalalignment='center', horizontalalignment='center',
              fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.suptitle('Huber LASSO + Screening DPP — Détection de sources\n'
                 f'{M} capteurs × {n_grid} directions, {k_actual} sources, bruit non-gaussien',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.savefig('array_dpp_well_conditioned.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'=' * 70}")
    print(f"CONCLUSION")
    print(f"{'=' * 70}")
    print(f"• Cohérence mutuelle : {mutual_coh:.4f} → "
          + ("BONNE" if mutual_coh < 0.5 else "MOYENNE" if mutual_coh < 0.9 else "MAUVAISE"))
    print(f"• DPP élimine {eliminated_avg:.0f} directions ({100 * eliminated_avg / n_grid:.1f}%)")
    print(
        f"• Détection : {tp_opt}/{k_actual} sources | Préc={prec_opt:.1f}% | Rappel={rec_opt:.1f}% | F1={best_f1:.3f}")