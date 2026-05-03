import numpy as np
from scipy.linalg import cho_factor, cho_solve

from model.RegressionAlgorithms.admm_lasso import admm_lasso_boyd


# ---------------------------
# Huber weights (IRLS)
# ---------------------------
def huber_weights(r, delta):
    abs_r = np.abs(r)
    w = np.ones_like(r)
    mask = abs_r > delta
    w[mask] = delta / abs_r[mask]
    return w


def admm_huber_adaptive_enet(X, y, beta_init = None,
                             lambda1=1.0, lambda2=0.1,
                             delta=1.345, rho=0.5, gamma=1,
                             max_admm_iter=100,
                             max_irls_iter=20,
                             tol=1e-4,
                             verbose=False
                             ):
    """
    Solve:
    min_beta sum Huber(y - X beta)
            + lambda1 * sum w_j |beta_j|
            + (lambda2/2) * ||beta||^2

    via ADMM + IRLS
    """

    n, p = X.shape

    abs_tol = 1e-4
    rel_tol = 1e-2

    mu = 1
    tau_incr = 2
    tau_decr = 2

    # ---------------------------
    # Initialisation
    # ---------------------------
    if beta_init is None:
        beta, _ = admm_lasso_boyd(X, y, lam=lambda1)
    else:
        beta = beta_init
        print('ok')

    w_adapt = 1.0 / (np.abs(beta) ** gamma + 1e-5)

    z = np.zeros(p)
    u = np.zeros(p)

    # Precompute
    Xt = X.T

    for k in range(max_admm_iter):

        # =========================
        # 1. BETA UPDATE (IRLS)
        # =========================
        v = z - u

        beta_old = beta.copy()

        for _ in range(max_irls_iter):

            r = y - X @ beta
            w = huber_weights(r, delta)

            # Weighted least squares
            WX = X * w[:, None]
            A = 2 * (Xt @ WX) + rho * np.eye(p)
            b = 2 * Xt @ (w * y) + rho * v

            # --- Cholesky solve ---
            L, lower = cho_factor(A, overwrite_a=False, check_finite=False)
            beta = cho_solve((L, lower), b, check_finite=False)

        # =========================
        # 2. Z UPDATE (Adaptive ENet)
        # =========================
        v = beta + u

        z_old = z.copy()

        z = soft_threshold(rho * v, lambda1 * w_adapt) / (rho + lambda2)

        # =========================
        # 3. DUAL UPDATE
        # =========================
        u = u + beta - z

        # =========================
        # 4. Adaptive weights update
        # =========================
        w_adapt = 1.0 / (np.abs(beta) ** gamma + 1e-8)

        # =========================
        # 5. Convergence check (Boyd)
        # =========================
        r_norm = np.linalg.norm(beta - z)
        s_norm = np.linalg.norm(-rho * (z - z_old))

        eps_pri = np.sqrt(p) * abs_tol + rel_tol * max(np.linalg.norm(beta), np.linalg.norm(z))
        eps_dual = np.sqrt(p) * abs_tol + rel_tol * np.linalg.norm(rho * u)

        if verbose:
            print(f"Iter {k} | r={r_norm:.3e} | s={s_norm:.3e}")

        # =========================
        # 6. ADAPTIVE RHO (Boyd)
        # =========================
        rho_old = rho

        if r_norm > mu * s_norm:
            rho *= tau_incr
        elif s_norm > mu * r_norm:
            rho /= tau_decr

        # Rescale dual variable
        if rho != rho_old:
            u *= (rho_old / rho)

        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return beta
# ---------------------------
# Soft-thresholding
# ---------------------------


def soft_threshold(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
# ---------------------------
# Main ADMM solver
# ---------------------------