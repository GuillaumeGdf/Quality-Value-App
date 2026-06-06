import cvxpy as cp

def huber_lasso_cvxpy(A, b, lambda_, delta, solver='CLARABEL', verbose=False):
    """
    Solve Huber-LASSO using CVXPY's interior-point solver.
    minimize sum(huber(Ax - b, M=delta)) + lambda_ * ||x||_1
    Returns optimal x, and objective value.
    """
    n = A.shape[1]
    x = cp.Variable(n)
    # Huber loss: cp.huber(x, M) with M the threshold parameter
    loss = cp.sum(cp.huber(A @ x - b, delta))
    reg = lambda_ * cp.norm1(x)
    objective = cp.Minimize(loss + reg)

    prob = cp.Problem(objective)
    prob.solve(solver=solver, verbose=verbose)
    return x.value, prob.value