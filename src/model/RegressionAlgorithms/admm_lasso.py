import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve


# ----------------------------
# Shrinkage (soft-threshold)
# ----------------------------
def shrinkage(x, kappa):
    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)


# ----------------------------
# Objective function
# ----------------------------
def objective(A, b, lam, x, z):
    return 0.5 * np.sum((A @ x - b) ** 2) + lam * np.linalg.norm(z, 1)


# ----------------------------
# Factorization (Boyd style)
# ----------------------------
def factor(A, rho):
    m, n = A.shape

    if m >= n:
        # skinny case
        M = A.T @ A + rho * np.eye(n)
        L, lower = cho_factor(M)
        return (L, lower, True)
    else:
        # fat case
        M = np.eye(m) + (1.0 / rho) * (A @ A.T)
        L, lower = cho_factor(M)
        return (L, lower, False)


# ----------------------------
# ADMM LASSO (Boyd version)
# ----------------------------
def admm_lasso_boyd(A, b, lam, rho=1.0, alpha=1.0,
                    max_iter=1000, abstol=1e-4, reltol=1e-2, verbose=False):

    m, n = A.shape

    abs_tol = 1e-4
    rel_tol = 1e-2

    mu = 5
    tau_incr = 2
    tau_decr = 2

    # Precompute
    Atb = A.T @ b

    # Init
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    # Factorization
    L, lower, skinny = factor(A, rho)

    history = {
        "objval": [],
        "r_norm": [],
        "s_norm": [],
        "eps_pri": [],
        "eps_dual": []
    }

    for k in range(max_iter):

        # ----------------------------
        # x-update
        # ----------------------------
        q = Atb + rho * (z - u)

        if skinny:
            # solve (A^T A + rho I)x = q
            x = cho_solve((L, lower), q)
        else:
            # Woodbury trick
            Aq = A @ q
            tmp = cho_solve((L, lower), Aq)
            x = (q / rho) - (A.T @ tmp) / (rho**2)

        # ----------------------------
        # z-update (over-relaxation)
        # ----------------------------
        zold = z.copy()

        x_hat = alpha * x + (1 - alpha) * zold
        z = shrinkage(x_hat + u, lam / rho)

        # ----------------------------
        # u-update
        # ----------------------------
        u = u + (x_hat - z)

        # ----------------------------
        # Diagnostics
        # ----------------------------
        r_norm = norm(x - z)
        s_norm = norm(-rho * (z - zold))

        eps_pri = np.sqrt(n) * abstol + reltol * max(norm(x), norm(z))
        eps_dual = np.sqrt(n) * abstol + reltol * norm(rho * u)

        obj = objective(A, b, lam, x, z)

        history["objval"].append(obj)
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        if verbose:
            print(f"{k:3d} | r={r_norm:.3e} | s={s_norm:.3e} | obj={obj:.4f}")

        # =========================
        # 6. ADAPTIVE RHO (Boyd)
        # =========================
        rho_old = rho

        if r_norm > mu * s_norm:
            rho *= tau_incr
            L, lower, skinny = factor(A, rho)

        elif s_norm > mu * r_norm:
            rho /= tau_decr
            L, lower, skinny = factor(A, rho)

        # Rescale dual variable
        if rho != rho_old:
            u *= (rho_old / rho)

        if r_norm < eps_pri and s_norm < eps_dual:
            break

    plt.figure()
    plt.plot(history["objval"])
    plt.title("Convergence objective")

    return x, history