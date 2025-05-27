import numpy as np
from utils import soft_threshold, sigmoid

def sigmoid(z):
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out

def soft_threshold(x, threshold):
    return np.sign(x) * max(abs(x) - threshold, 0.0)

def acd_logreg_l1(X, y, lambd=1e-3, epochs=50, seed=0):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    L = 0.25 * (X ** 2).sum(axis=0) / n_samples # Lipschitz constant
    L[L == 0] = 1e-12

    w = np.zeros(n_features)
    w_prev = w.copy()
    t_prev = 1.0
    z = X @ w  
    loss_hist = []
    def objective():
        z = X @ w
        return np.mean(np.log1p(np.exp(z)) - y * z) + lambd * np.linalg.norm(w, 1)

    # def objective():
    #     return np.mean(np.log1p(np.exp(-y * z))) + lambd * np.linalg.norm(w, 1)

    loss_hist.append(objective())

    for _ in range(epochs):
        coords = rng.permutation(n_features)

        # Nesterov acceleration
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
        v = w + (t_prev - 1) / t * (w - w_prev) # extrapolated point
        z = X @ v

        for j in coords:
            residual = sigmoid(z) - y 
            g_j = residual @ X[:, j] / n_samples # j-ta składowa gradientu log-loss w punkcie v
            alpha = 1.0 / L[j] 
            w_new_j = soft_threshold(v[j] - alpha * g_j, lambd * alpha)
            delta = w_new_j - v[j]
            if delta:
                v[j] = w_new_j
                z += delta * X[:, j]

        w_prev, w, t_prev = w, v.copy(), t
        loss_hist.append(objective())

    return w, loss_hist


def cd_logreg_l1(X, y, lambd=1e-3, epochs=50, seed=0):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    L = 0.25 * (X**2).sum(axis=0) / n_samples
    L[L == 0] = 1e-12
    w = np.zeros(n_features)
    z = X @ w          
    loss_hist = []
    # def objective():
    #     return np.mean(np.log1p(np.exp(-y * z))) + lambd * np.linalg.norm(w, 1)
    def objective():
        z = X @ w
        return np.mean(np.log1p(np.exp(z)) - y * z) + lambd * np.linalg.norm(w, 1)


    loss_hist.append(objective())
    for _ in range(epochs):
        coords = rng.permutation(n_features)    

        for j in coords:
            residual = sigmoid(z) - y
            g_j = residual @ X[:, j] / n_samples  
            alpha = 1.0 / L[j]                      
            w_new = soft_threshold(w[j] - alpha * g_j, lambd * alpha)

            delta = w_new - w[j]
            if delta:                             
                w[j] = w_new
                z += delta * X[:, j]           

        loss_hist.append(objective())

    return w, loss_hist