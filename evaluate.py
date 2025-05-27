import numpy as np
import matplotlib.pyplot as plt
from fista import fista_logreg_l1
from cd import cd_logreg_l1, acd_logreg_l1
from sklearn.metrics import accuracy_score, f1_score

def plot_our_history(X, y, epochs=50, lam=1e-3):
    w_acc, hist_acc = acd_logreg_l1(X, y, lambd=lam, epochs=epochs, seed=42)
    w_plain, hist_plain = cd_logreg_l1(X, y, lambd=lam, epochs=epochs, seed=42)
    w_fista, hist_fista = fista_logreg_l1(X, y, lambd=lam, epochs=epochs)
    
    plt.plot(hist_plain, label="CD", linestyle="--")
    plt.plot(hist_acc,   label="Accelerated CD")
    plt.plot(hist_fista, label="FISTA", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.legend()
    plt.show()
    
    y_pred_acc   = (X @ w_acc >= 0).astype(int)
    y_pred_plain = (X @ w_plain >= 0).astype(int)
    y_pred_fista = (X @ w_fista >= 0).astype(int)
    
    print("Accuracy:")
    print(f"Accelerated CD: {accuracy_score(y, y_pred_acc):.3f}")
    print(f"CD: {accuracy_score(y, y_pred_plain):.3f}")
    print(f"FISTA: {accuracy_score(y, y_pred_fista):.3f}")
    
    print("F1 score:")
    print(f"Accelerated CD: {f1_score(y, y_pred_acc):.3f}")
    print(f"CD: {f1_score(y, y_pred_plain):.3f}")
    print(f"FISTA: {f1_score(y, y_pred_fista):.3f}")

    