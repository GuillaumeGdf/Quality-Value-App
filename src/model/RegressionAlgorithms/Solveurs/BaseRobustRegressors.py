from typing import Optional

import numpy as np
import cvxpy as cp

from abc import ABC, abstractmethod
from enum import Enum
from numpy.typing import NDArray

class SolverType(Enum):
    CVXPY_HUBER_LASSO = "cvxpy_huber_lasso"
    ADMM_HUBER_LASSO = "admm_huber_lasso"
    CVXPY_ADAPTIVE_HUBER_LASSO = "cvxpy_adaptive_huber_lasso"
    ADMM_ADAPTIVE_HUBER_LASSO = "admm_adaptive_huber_lasso"


class BaseRobustRegressor(ABC):
    """
    Interface unifiée pour les algorithmes de régression robuste du screener.
    Gère la standardisation et l'isolement de l'intercept.
    """

    def __init__(self, lambda_: float = 1.0, delta: float = 1.35, fit_intercept: bool = True):
        self.lambda_ = lambda_
        self.delta = delta
        self.fit_intercept = fit_intercept
        self.coef_: Optional[NDArray[float]] = None
        self.intercept_: float = 0.0

    @abstractmethod
    def _fit_core(self, A: NDArray[float], b: NDArray[float], penalty_weights: NDArray[float]):
        """Méthode interne à implémenter par chaque solveur spécifique."""
        pass

    def fit(self, X: NDArray[float], y: NDArray[float], initial_weights: Optional[NDArray[float]] = None):
        """
        Prépare les données, gère l'intercept et exécute le solveur.
        """
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float).flatten()
        n_samples, n_features = A.shape

        # Initialisation des poids pour le LASSO (vecteur de 1 par défaut)
        if initial_weights is None:
            penalty_weights = np.ones(n_features)
        else:
            penalty_weights = np.asarray(initial_weights, dtype=float)

        if self.fit_intercept:
            # Pour éviter de pénaliser l'intercept, on ajoute une colonne de 1 à A
            # et on ajoute un poids de 0 dans le vecteur de pénalité pour cette colonne.
            A_comp = np.hstack([np.ones((n_samples, 1)), A])
            weights_comp = np.concatenate(([0.0], penalty_weights))
        else:
            A_comp = A
            weights_comp = penalty_weights

        # Appel du cœur du solveur
        x_opt = self._fit_core(A_comp, b, weights_comp)

        # Extraction des coefficients et de l'intercept
        if self.fit_intercept:
            self.intercept_ = x_opt[0]
            self.coef_ = x_opt[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = x_opt

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Le modèle n'a pas encore été ajusté (fit).")
        return np.dot(X, self.coef_) + self.intercept_


class CVXPYHuberLassoStrategy(BaseRobustRegressor):
    """Implémentation native de votre code CVXPY personnalisé."""

    def __init__(self, lambda_=1.0, delta=1.35, fit_intercept=True, solver: str = 'CLARABEL', verbose: bool = False):
        super().__init__(lambda_, delta, fit_intercept)
        self.solver = solver
        self.verbose = verbose

    def _fit_core(self, A: NDArray[float], b: NDArray[float], penalty_weights: NDArray[float]):
        n = A.shape[1]
        x = cp.Variable(n)

        # Perte de Huber
        loss = cp.sum(cp.huber(A @ x - b, self.delta))

        # Pénalité L1 pondérée (gère le Lasso standard et l'Adaptive grâce à penalty_weights)
        # cp.multiply effectue un produit élément par élément
        reg = self.lambda_ * cp.norm1(cp.multiply(penalty_weights, x))

        objective = cp.Minimize(loss + reg)
        prob = cp.Problem(objective)
        prob.solve(solver=self.solver, verbose=self.verbose)

        return x.value


class ADMMHuberLassoStrategy(BaseRobustRegressor):
    """Implémentation de votre code ADMM optimisé."""

    def __init__(self, lambda_: float = 1.0, delta: float = 1.35,
                 fit_intercept: bool = True, rho: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-6,
                 verbose: bool = False):
        super().__init__(lambda_, delta, fit_intercept)
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def __huber_prox_standard(self, v):
        """
        Opérateur proximal NumPy pour la fonction de Huber statistique standard.

        Formulation mathématique de la perte :
            h(t) = 0.5 * t^2          si |t| <= delta
                   delta * (|t| - 0.5 * delta)  si |t| > delta

        Ce problème résout : min_t { h(t) + (rho / 2) * (t - v)^2 }

        Seuil de transition : delta * (1 + rho)
        """
        threshold = self.delta * (1 + self.rho)
        t = np.where(
            np.abs(v) <= threshold,
            v / (1 + self.rho),
            v - self.rho * self.delta * np.sign(v)
        )
        return t

    def __huber_prox_cvxpy_scaled(self, v):
        """
        Opérateur proximal NumPy calibré sur la fonction de Huber de CVXPY.

        Formulation mathématique de la perte (CVXPY) :
            h_cvxpy(t) = t^2                    si |t| <= delta
                         2 * delta * |t| - delta^2   si |t| > delta

        Note : Cette version n'intègre pas le facteur 0.5 dans la zone quadratique,
        ce qui multiplie par 2 la pente de la zone linéaire par rapport à la version standard.

        Ce problème résout : min_t { h_cvxpy(t) + (rho / 2) * (t - v)^2 }

        Seuil de transition dérivé : delta * (1 + 2.0 / rho)
        """
        threshold = self.delta * (1 + 2.0 / self.rho)
        t_quad = v / (1 + 2.0 / self.rho)
        t_lin = v - (2.0 * self.delta / self.rho) * np.sign(v)
        return np.where(np.abs(v) <= threshold, t_quad, t_lin)

    def __soft_threshold(self, v: NDArray[float], kappa: NDArray[float] | float):
        """
        Opérateur Proximal associé à la norme L1 (Soft-Thresholding).

        Résout le problème d'optimisation suivant pour chaque composante :
        argmin_z { kappa * ||z||_1 + 0.5 * ||z - v||_2^2 }

        Cette fonction applique un seuillage doux élément par élément. Elle réduit
        la magnitude des composants de v vers zéro d'une valeur kappa, et annule
        les composants dont la magnitude est inférieure à kappa.

        Args:
            v (NDArray[float]): Vecteur ou matrice des coefficients d'entrée à seuiller
                                (typiquement w2 = x_new + u2 dans l'algorithme ADMM).
            kappa (NDArray[float] ou float): Le seuil de pénalisation. Dans le cadre
                                             de l'Adaptive LASSO, il s'agit d'un vecteur
                                             contenant (lambda_ * penalty_weights) / rho.

        Returns:
            NDArray[float]: Le vecteur seuillé de même dimension que v, où les variables
                            non significatives sont fixées exactement à 0.0.
        """
        # kappa supporte nativement un scalaire ou un vecteur de taille (n,) grâce au broadcasting NumPy.
        # Cela permet de ne pas pénaliser l'intercept (kappa_0 = 0.0) et d'appliquer les poids adaptatifs.
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0)

    def _fit_core(self, A: NDArray[float], b: NDArray[float], penalty_weights: NDArray[float]):
        m, n = A.shape
        ATA_plus_I = A.T @ A + np.eye(n)
        L = np.linalg.cholesky(ATA_plus_I)

        x = np.zeros(n)
        z1 = np.zeros(m)
        z2 = np.zeros(n)
        u1 = np.zeros(m)
        u2 = np.zeros(n)

        # Pré-calcul du vecteur de seuillage pour chaque composante
        kappa = (self.lambda_ * penalty_weights) / self.rho

        for k in range(self.max_iter):
            v1 = z1 + b - u1
            v2 = z2 - u2
            rhs = A.T @ v1 + v2
            x_new = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

            w1 = A @ x_new - b + u1
            z1_new = self.__huber_prox_cvxpy_scaled(w1)

            w2 = x_new + u2
            # Utilisation du kappa vectoriel (gère l'absence de pénalité sur l'intercept)
            z2_new = self.__soft_threshold(w2, kappa)

            u1_new = u1 + A @ x_new - z1_new - b
            u2_new = u2 + x_new - z2_new

            primal_res = np.linalg.norm(A @ x_new - z1_new - b) + np.linalg.norm(x_new - z2_new)
            dual_res = self.rho * np.linalg.norm(A.T @ (z1_new - z1) + (z2_new - z2))

            x, z1, z2, u1, u2 = x_new, z1_new, z2_new, u1_new, u2_new

            if primal_res < self.tol and dual_res < self.tol:
                if self.verbose:
                    print(f"ADMM Converged at iteration {k}")
                break

        return x