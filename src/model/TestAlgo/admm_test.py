import numpy as np
import matplotlib.pyplot as plt

from model.RegressionAlgorithms.admm_lasso import admm_lasso_boyd
from model.RegressionAlgorithms.admm_huber_adaptative_elastic_net import admm_huber_adaptive_enet

np.random.seed(42)

# Dimensions
n = 300     # observations
p = 20     # dictionnaire (surcomplet)
k = 10       # nb de sources réelles

# -----------------------------
# Dictionnaire corrélé
# -----------------------------
def generate_correlated_X(n, p, corr=0.9):
    X = np.random.randn(n, p)
    for j in range(1, p):
        X[:, j] = corr * X[:, j-1] + np.sqrt(1-corr**2) * X[:, j]
    return X / np.linalg.norm(X, axis=0)

X = generate_correlated_X(n, p, corr=0.70)

# -----------------------------
# Vraies sources (sparse)
# -----------------------------
beta_true = np.zeros(p)
support = np.random.choice(p, k, replace=False)
beta_true[support] = np.random.uniform(1, 2, size=k) * np.random.choice([-1,1], k)

# -----------------------------
# Bruit
# -----------------------------
noise = 0.05 * np.random.randn(n)

# Bruit impulsionnel (outliers)
size=10
outliers = np.zeros(n)
idx_out = np.random.choice(n, size=size, replace=False)
outliers[idx_out] = np.random.uniform(2, 5, size=size)

# Signal
y = X @ beta_true + noise + outliers

lambda1 = 0.1
lambda2 = 0.2

beta_init = admm_huber_adaptive_enet(X, y, beta_init=np.ones(X.shape[1]), lambda1=0.8*lambda1, lambda2=lambda2, max_admm_iter=5)
beta_huber = admm_huber_adaptive_enet(X, y, beta_init=beta_init, lambda1=lambda1, lambda2=lambda2)
beta_lasso, _ = admm_lasso_boyd(X, y, lam=lambda1)


def support(x, tol=1e-3):
    return set(np.where(np.abs(x) > tol)[0])

# true_support = set(support)
# lasso_support = support(beta_lasso)
# huber_support = support(beta_huber)
#
# print("True:", true_support)
# print("LASSO:", lasso_support)
# print("Huber:", huber_support)

fig, ax = plt.subplots(1, 2)
ax[0].plot(beta_true, label="True", linewidth=3)
ax[0].plot(beta_lasso, '--', label="LASSO")

ax[1].plot(beta_true, label="True", linewidth=3)
ax[1].plot(beta_huber, '--', label="Huber Adaptive ENet")

ax[0].legend()
ax[1].legend()
plt.title("Reconstruction des sources")
plt.show()
