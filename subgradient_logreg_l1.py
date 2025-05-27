import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid function."""
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out

def subgradient_logreg_l1(X, y, lambd=1e-3, epochs=100, learning_rate=1e-3, 
                          step_type='diminishing', decay_rate=0.95, seed=0):
    """
    Subgradient method for L1-regularized logistic regression.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target values, should be in {-1, 1}.
    lambd : float, default=1e-3
        Regularization strength.
    epochs : int, default=100
        Maximum number of passes over the training data.
    learning_rate : float, default=0.01
        Initial learning rate.
    step_type : str, default='constant'
        Type of step size schedule:
        - 'constant': Fixed step size
        - 'diminishing': Step size of form learning_rate/(epoch+1)
        - 'exponential': Step size decays exponentially as learning_rate * decay_rate^epoch
    decay_rate : float, default=0.95
        Learning rate decay factor (used for exponential decay).
    seed : int, default=0
        Random seed for reproducibility.
        
    Returns:
    --------
    w : ndarray of shape (n_features,)
        Coefficient vector.
    loss_hist : list
        Loss values at each epoch.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    z = X @ w
    loss_hist = []
    
    # Define objective function
    def objective():
        return np.mean(np.log1p(np.exp(-y * z))) + lambd * np.linalg.norm(w, 1)
    
    loss_hist.append(objective())
    
    base_lr = learning_rate
    
    for epoch in range(epochs):
        # Compute step size based on selected strategy
        if step_type == 'constant':
            lr = base_lr
        elif step_type == 'diminishing':
            # Classical diminishing step size 
            lr = 1 / (epoch + 1)
        elif step_type == 'exponential':
            # Exponential decay
            lr = base_lr * (decay_rate ** epoch)
        else:
            raise ValueError(f"Unknown step_type: {step_type}")
        
        # Compute gradient of logistic loss
        probs = sigmoid(y * z)
        grad_loss = -X.T @ ((1 - probs) * y) / n_samples
        
        # Compute subgradient of L1 regularization term (-1, 0, 1)
        subgrad_l1 = np.zeros_like(w)
        nonzero_mask = w != 0
        # Non-zero weights contribute their sign
        subgrad_l1[nonzero_mask] = np.sign(w[nonzero_mask])
        zero_mask = w == 0
        # For zero weights, randomly assign a subgradient in [-1, 1]
        subgrad_l1[zero_mask] = rng.uniform(-1, 1, size=np.sum(zero_mask))
        subgrad_l1 = lambd * subgrad_l1
        
        # Full subgradient
        subgrad = grad_loss + subgrad_l1
        
        # Update weights
        w = w - lr * subgrad
        
        # Update predictions
        z = X @ w
        
        # Track loss
        loss_hist.append(objective())
    
    return w, loss_hist