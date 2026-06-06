import numpy as np


def huber_prox_cvxpy(v, rho, delta):
    """
    Proximal operator of (1/rho) * h_CVXPY(·, delta) where
        h_CVXPY(t) = t^2                     if |t| <= delta
                      2*delta*|t| - delta^2  if |t| > delta.
    Minimizes h_CVXPY(t) + (rho/2)*(t - v)^2.
    """
    # Seuil de transition déterminé par la condition de rester dans la zone quadratique.
    # Solution quadratique : t = v / (1 + 2/rho)   car h_CVXPY(t) = t^2 -> dérivée 2t
    # Condition |t| <= delta  => |v| <= delta * (1 + 2/rho)
    threshold = delta * (1 + 2.0 / rho)
    t_quad = v / (1 + 2.0 / rho)

    # Solution linéaire : pour t > 0, dérivée = 2*delta + rho*(t - v) = 0 => t = v - 2*delta/rho
    t_lin = v - (2.0 * delta / rho) * np.sign(v)

    return np.where(np.abs(v) <= threshold, t_quad, t_lin)


def huber_prox(v, rho, delta):
    """
    Proximal operator of (1/rho) * h_delta (element-wise).
    h_delta(t) = 0.5 t^2 if |t| <= delta else delta(|t| - 0.5 delta)
    The proximal step solves min_t h_delta(t) + (rho/2)*(t - v)^2.
    Closed form:
    if |v| <= delta*(1+rho): t = v / (1+rho)
    else: t = v - rho*delta*sign(v)
    """
    threshold = delta * (1 + rho)
    t = np.where(np.abs(v) <= threshold,
                 v / (1 + rho),
                 v - rho * delta * np.sign(v))
    return t

def soft_threshold(v, kappa):
    """Element-wise soft thresholding."""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)

def huber_lasso_admm(A, b, lambda_, delta, rho=1.0, max_iter=1000, tol=1e-6, verbose=False):
    """
    Solve Huber-LASSO: minimize H_delta(Ax - b) + lambda_ * ||x||_1
    via ADMM with two auxiliary variables:
        z1 = A x - b
        z2 = x
    Returns:
        x_opt, history (dict with residuals)
    """
    m, n = A.shape
    # Precompute matrix for x-update: (A^T A + I)^{-1}
    # We solve (A^TA + I) x = rhs, using Cholesky.
    ATA_plus_I = A.T @ A + np.eye(n)
    # Cholesky factorization
    L = np.linalg.cholesky(ATA_plus_I)  # lower triangular

    # Initialization
    x = np.zeros(n)
    z1 = np.zeros(m)
    z2 = np.zeros(n)
    u1 = np.zeros(m)
    u2 = np.zeros(n)

    history = {
        'primal_residual': [],
        'dual_residual': [],
        'objective': []
    }

    for k in range(max_iter):
        # --- x-update ---
        v1 = z1 + b - u1
        v2 = z2 - u2
        rhs = A.T @ v1 + v2
        # Solve L L^T x = rhs
        x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))  # or use cho_solve

        # --- z1-update (Huber proximal) ---
        w1 = A @ x_new - b + u1
        z1_new = huber_prox_cvxpy(w1, rho, delta)

        # --- z2-update (soft threshold) ---
        w2 = x_new + u2
        z2_new = soft_threshold(w2, lambda_ / rho)

        # --- dual update ---
        u1_new = u1 + A @ x_new - z1_new - b
        u2_new = u2 + x_new - z2_new

        # Residuals for convergence
        primal_res = np.linalg.norm(A @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
        dual_res = rho * (np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2)))
        # dual residual formula for scaled form: rho * ||A^T(u_new - u)||? We can approximate.
        # Standard ADMM residuals: primal = ||Ax - z - c||_2, dual = rho * ||A^T B (z_new - z)||_2
        # For our problem: B = -I, so dual = rho * || (A^T(z1_new-z1)) + (z2_new-z2) ||_2
        # Let's compute properly:
        dual_res = rho * np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2))

        # Update variables
        x, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

        # Track objective: H_delta(Ax-b) + lambda_*||x||_1
        resid = A @ x - b
        huber_loss = np.sum(np.where(np.abs(resid) <= delta,
                                     resid ** 2,
                                     2 * delta * np.abs(resid) - delta ** 2))
        obj = huber_loss + lambda_ * np.sum(np.abs(x))
        history['primal_residual'].append(primal_res)
        history['dual_residual'].append(dual_res)
        history['objective'].append(obj)

        if verbose and k % 100 == 0:
            print(f"ADMM iter {k}: primal res {primal_res:.2e}, dual res {dual_res:.2e}, obj {obj:.6f}")

        if primal_res < tol and dual_res < tol:
            print(f"Converged at iteration {k}")
            break

    return x, history