import numpy as np

def sigmoid(z):
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def soft_threshold_vec(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def fista_logreg_l1(X, y, lambd=1e-3, epochs=100):
    n_samples, n_features = X.shape
    y = np.where(y <= 0, -1.0, 1.0)

    # max eigenvalue (1/4 * (X^T *X)) / n
    L = 0.25 * (np.linalg.norm(X, ord=2) ** 2) / n_samples
    if L == 0:
        L = 1e-12  

    w = np.zeros(n_features)
    w_prev = w.copy()
    t_prev = 1.0

    def objective(w_vec):
        z_vec = X @ w_vec
        return np.mean(np.log1p(np.exp(-y * z_vec))) + lambd * np.linalg.norm(
            w_vec, 1
        )

    loss_hist = [objective(w)]

    for _ in range(epochs):
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
        y_k = w + (t_prev - 1) / t * (w - w_prev)

        z = X @ y_k
        residual = sigmoid(z) - (y + 1) / 2  
        grad = X.T @ residual / n_samples

        w_next = soft_threshold_vec(y_k - grad / L, lambd / L)

        w_prev, w, t_prev = w, w_next, t
        loss_hist.append(objective(w))

    return w, loss_hist


