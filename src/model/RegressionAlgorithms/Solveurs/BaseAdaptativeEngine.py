import numpy as np

from model.RegressionAlgorithms.Solveurs.BaseRobustRegressors import CVXPYHuberLassoStrategy, \
            ADMMHuberLassoStrategy, SolverType


class HuberAdaptiveLassoOrchestrator:
    """
    Orchestrateur qui exécute l'Adaptive Lasso en 2 étapes :
    1. Régression Huber pure pour obtenir les coefficients initiaux.
    2. Calcul des poids adaptatifs : w_j = 1 / |beta_j|^gamma
    3. Régression Huber Lasso finale avec les poids calculés.
    """

    def __init__(self, strategy_type: SolverType, lambda_=1.0, delta=1.35, gamma=1.0, fit_intercept=True):
        self.strategy_type = strategy_type
        self.lambda_ = lambda_
        self.delta = delta
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.model_ = None

    def fit(self, X, y):
        # Étape 1 : Instancier un modèle sans pénalité (lambda=0) pour avoir les coefficients de départ
        if "cvxpy" in self.strategy_type.value:
            first_stage = CVXPYHuberLassoStrategy(lambda_=0.0, delta=self.delta, fit_intercept=self.fit_intercept)
            final_strategy = CVXPYHuberLassoStrategy(lambda_=self.lambda_, delta=self.delta,
                                                     fit_intercept=self.fit_intercept)
        else:
            first_stage = ADMMHuberLassoStrategy(lambda_=0.0, delta=self.delta, fit_intercept=self.fit_intercept)
            final_strategy = ADMMHuberLassoStrategy(lambda_=self.lambda_, delta=self.delta,
                                                    fit_intercept=self.fit_intercept)

        # Ajustement de la première étape
        first_stage.fit(X, y)
        beta_init = first_stage.coef_

        # Étape 2 : Calcul des poids adaptatifs (avec une sécurité numérique pour éviter la division par zéro)
        eps = 1e-4
        penalty_weights = 1.0 / (np.abs(beta_init) + eps) ** self.gamma

        # Étape 3 : Ajustement final avec les poids calculés
        self.model_ = final_strategy.fit(X, y, initial_weights=penalty_weights)
        self.coef_ = final_strategy.coef_
        self.intercept_ = final_strategy.intercept_

    def predict(self, X):
        return self.model_.predict(X)


class RobustRegressorFactory:
    """Factory permettant de basculer d'un solveur à un autre via une simple variable."""

    @staticmethod
    def create_solver(solver_type: SolverType, lambda_=1.0, delta=1.35, fit_intercept=True, **kwargs):
        if solver_type == SolverType.CVXPY_HUBER_LASSO:
            return CVXPYHuberLassoStrategy(lambda_, delta, fit_intercept, **kwargs)

        elif solver_type == SolverType.ADMM_HUBER_LASSO:
            return ADMMHuberLassoStrategy(lambda_, delta, fit_intercept, **kwargs)

        elif solver_type in [SolverType.CVXPY_ADAPTIVE_HUBER_LASSO, SolverType.ADMM_ADAPTIVE_HUBER_LASSO]:
            gamma = kwargs.pop('gamma', 1.0)
            return HuberAdaptiveLassoOrchestrator(solver_type, lambda_, delta, gamma, fit_intercept)

        else:
            raise ValueError(f"Solver type {solver_type} non reconnu.")