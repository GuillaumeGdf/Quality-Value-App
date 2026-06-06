import numpy as np
import matplotlib.pyplot as plt
import time
from model.RegressionAlgorithms.Solveurs.ADMM.ADMM_HUBER_LASSO import huber_lasso_admm
from model.RegressionAlgorithms.Solveurs.PointInterieurs.IPM_HUBER_LASSO import huber_lasso_cvxpy

# ========== Génération des données synthétiques ==========
np.random.seed(42)
m = 500         # observations
n = 15          # variables explicatives
k_true = 5      # nombre de coefficients non nuls
outlier_ratio = 0.1  # proportion d'outliers

# Matrice de design A (colonnes peu corrélées)
# On génère des variables normales indépendantes, puis on ajoute une légère corrélation via une structure Toeplitz
cov = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cov[i, j] = 0.5 ** abs(i - j)   # corrélation décroissante
A = np.random.multivariate_normal(np.zeros(n), cov, size=m)

# Vrai vecteur x_true (parcimonieux)
x_true = np.zeros(n)
true_indices = np.random.choice(n, k_true, replace=False)
x_true[true_indices] = np.random.randn(k_true) * 2.0

# Bruit : mélange de bruit gaussien et d'outliers
noise = np.random.randn(m) * 0.5
outlier_mask = np.random.rand(m) < outlier_ratio
noise[outlier_mask] += np.random.randn(np.sum(outlier_mask)) * 10.0  # grandes valeurs
b = A @ x_true + noise

print(f"Vrai x (non nuls): {x_true[x_true != 0]}")

# Paramètres du modèle
lambda_reg = 1.0
huber_delta = 1.345   # seuil de robustesse

# ========== Résolution avec ADMM ==========
print("\n--- ADMM ---")
start_admm = time.time()
x_admm, hist = huber_lasso_admm(A, b, lambda_reg, huber_delta, rho=1.0,
                                max_iter=2000, tol=1e-8, verbose=False)
time_admm = time.time() - start_admm

# ========== Résolution avec CVXPY ==========
print("\n--- CVXPY ---")
start_cvx = time.time()
x_cvx, obj_cvx = huber_lasso_cvxpy(A, b, lambda_reg, huber_delta, solver='CLARABEL', verbose=False)
time_cvx = time.time() - start_cvx

# ========== Comparaison ==========
# Objectifs finaux (vérification)
resid_admm = A @ x_admm - b
huber_admm = np.sum(np.where(np.abs(resid_admm) <= huber_delta,
                                     resid_admm ** 2,
                                     2 * huber_delta * np.abs(resid_admm) - huber_delta ** 2))
obj_admm = huber_admm + lambda_reg * np.linalg.norm(x_admm, 1)

print("\nRésultats:")
print(f"ADMM - objective: {obj_admm:.6f}, time: {time_admm:.4f}s")
print(f"CVXPY - objective: {obj_cvx:.6f}, time: {time_cvx:.4f}s")
print(f"Difference ||x_admm - x_cvx||_2 = {np.linalg.norm(x_admm - x_cvx):.2e}")
print(f"||x_admm - x_true||_2 = {np.linalg.norm(x_admm - x_true):.4f}")
print(f"||x_cvx - x_true||_2 = {np.linalg.norm(x_cvx - x_true):.4f}")

# ========== Graphiques ==========
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 1. Coefficients estimés vs vrai
ax = axes[0, 0]
ax.bar(np.arange(n) - 0.2, x_true, width=0.2, label='Vrai', color='green')
ax.bar(np.arange(n), x_admm, width=0.2, label='ADMM', color='blue')
ax.bar(np.arange(n) + 0.2, x_cvx, width=0.2, label='CVXPY', color='orange')
ax.set_title("Coefficients estimés")
ax.set_xlabel("Index")
ax.legend()

# 2. Différence entre solutions
ax = axes[0, 1]
ax.plot(x_admm - x_cvx, 'o-', color='red')
ax.set_title("Différence ADMM - CVXPY")
ax.set_xlabel("Index")
ax.grid(True)

# 3. Résidus ADMM
ax = axes[0, 2]
ax.semilogy(hist['primal_residual'], label='Primal')
ax.semilogy(hist['dual_residual'], label='Dual')
ax.set_title("Convergence ADMM (résidus)")
ax.set_xlabel("Itération")
ax.legend()
ax.grid(True)

# 4. Objectif ADMM au cours des itérations
ax = axes[1, 0]
ax.plot(hist['objective'])
ax.axhline(obj_cvx, color='r', linestyle='--', label='CVXPY objective')
ax.set_title("Évolution de l'objectif ADMM")
ax.set_xlabel("Itération")
ax.legend()
ax.grid(True)

# 5. Comparaison des prédictions (A x)
ax = axes[1, 1]
pred_admm = A @ x_admm
pred_cvx = A @ x_cvx
ax.scatter(b, pred_admm, alpha=0.4, label='ADMM', s=10)
ax.scatter(b, pred_cvx, alpha=0.4, label='CVXPY', s=10)
ax.plot([b.min(), b.max()], [b.min(), b.max()], 'k--')
ax.set_title("Prédictions vs Observations")
ax.set_xlabel("Observations b")
ax.set_ylabel("Prédictions Ax")
ax.legend()

# 6. Résidus (Ax - b) des deux méthodes
ax = axes[1, 2]
res_admm = A @ x_admm - b
res_cvx = A @ x_cvx - b
ax.hist(res_admm, bins=30, alpha=0.5, label='ADMM', color='blue')
ax.hist(res_cvx, bins=30, alpha=0.5, label='CVXPY', color='orange')
ax.set_title("Histogramme des résidus")
ax.legend()

plt.tight_layout()
plt.show()