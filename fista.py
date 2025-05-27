import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(X, y, w):
    z = X @ w
    return np.mean(np.log1p(np.exp(-y * z)))

def grad_logistic_loss(X, y, w):
    z = X @ w
    s = sigmoid(-y * z)
    return -(1 / X.shape[0]) * (X.T @ (y * s))

def soft_thresholding(x, lambd):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

def compute_lipschitz_constant(X):
    n_samples = X.shape[0]
    eigvals = np.linalg.eigvalsh(X.T @ X)
    L = 0.25 * np.max(eigvals) / n_samples
    return L


def fista_logreg_l1(X, y, lambd, epochs=100):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    z = w.copy()
    t = 1.0
    L = compute_lipschitz_constant(X)
    
    def objective(w):
        z = X @ w
        return np.mean(np.log1p(np.exp(-y * z))) + lambd * np.linalg.norm(w, 1)

    history = []
    for i in range(epochs):
        w_old = w.copy()
        
        grad = grad_logistic_loss(X, y, z)
        w = soft_thresholding(z - grad / L, lambd / L)
        
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = w + ((t - 1) / t_new) * (w - w_old)
        t = t_new
        history.append(objective(w))

    return w, history



