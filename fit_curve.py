"""Fits theta, M, X for the parametric curve. See README.md for full explanation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

DATA_PATH = "xy_data.csv"
T_MIN, T_MAX = 6.0, 60.0
BOUNDS = [(0, 50), (-0.05, 0.05), (0, 100)]


def curve_xy(t, theta_deg, M, X):
    theta = np.radians(theta_deg)
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - v * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + v * np.cos(theta)
    return x, y


def chamfer_objective(params, data_xy, n_samples):
    theta_deg, M, X = params
    t_samples = np.linspace(T_MIN, T_MAX, n_samples)
    cx, cy = curve_xy(t_samples, theta_deg, M, X)
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, _ = tree.query(data_xy, p=1)
    return np.mean(dist)


def fit(data_xy):
    de_result = differential_evolution(
        chamfer_objective, BOUNDS, args=(data_xy, 1500),
        maxiter=150, popsize=25, tol=1e-9,
        seed=42, workers=-1, updating="deferred", polish=False,
    )
    local_result = minimize(
        chamfer_objective, de_result.x, args=(data_xy, 20000),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 10000, "maxfev": 10000},
    )
    return local_result.x, local_result.fun


def to_latex(theta_deg, M, X):
    theta_rad = np.radians(theta_deg)
    return (
        f"\\left(t*\\cos({theta_rad:.4f})-e^{{{M:.4f}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\sin({theta_rad:.4f})+{X:.4g},42+t*\\sin({theta_rad:.4f})+"
        f"e^{{{M:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_rad:.4f})\\right)"
    )


def invert(theta_deg, X, data_xy):
    theta = np.radians(theta_deg)
    xs = data_xy[:, 0] - X
    ys = data_xy[:, 1] - 42
    t = xs * np.cos(theta) + ys * np.sin(theta)
    v = -xs * np.sin(theta) + ys * np.cos(theta)
    return t, v


def fit_M_linear(t, v):
    s = np.sin(0.3 * t)
    mask = np.abs(s) > 0.2
    ratio = v[mask] / s[mask]
    valid = ratio > 1e-6
    tt, rr = t[mask][valid], ratio[valid]
    if len(tt) < 5:
        return None
    L = np.log(rr)
    return np.sum(tt * L) / np.sum(tt * tt)


def objective_2d(params, data_xy):
    theta_deg, X = params
    t, v = invert(theta_deg, X, data_xy)
    M = fit_M_linear(t, v)
    if M is None or not (-0.05 <= M <= 0.05):
        return 1e12
    pred_v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return np.sum((v - pred_v) ** 2)


def fit_via_linear_reduction(data_xy):
    best = None
    for theta_deg in np.arange(0, 50.01, 1.0):
        for X in np.arange(0, 100.01, 2.0):
            val = objective_2d([theta_deg, X], data_xy)
            if best is None or val < best[0]:
                best = (val, theta_deg, X)

    res = minimize(objective_2d, [best[1], best[2]], args=(data_xy,),
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000})
    theta_deg, X = res.x
    t, v = invert(theta_deg, X, data_xy)
    M = fit_M_linear(t, v)
    return theta_deg, M, X, res.fun


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    data_xy = df[["x", "y"]].to_numpy()

    (theta_deg, M, X), residual = fit(data_xy)
    print("Method 1 (Chamfer / differential evolution):")
    print(f"  theta = {theta_deg:.6f} deg  ({np.radians(theta_deg):.6f} rad)")
    print(f"  M     = {M:.6f}")
    print(f"  X     = {X:.6f}")
    print(f"  mean L1 residual = {residual:.6f}")

    theta_deg2, M2, X2, res2 = fit_via_linear_reduction(data_xy)
    print("\nMethod 2 (rotation-inversion + linear regression, validation):")
    print(f"  theta = {theta_deg2:.6f} deg")
    print(f"  M     = {M2:.6f}")
    print(f"  X     = {X2:.6f}")
    print(f"  sum-of-squares residual = {res2:.10f}")

    t_fine = np.linspace(T_MIN, T_MAX, 50000)
    cx, cy = curve_xy(t_fine, theta_deg, M, X)
    tree = cKDTree(np.column_stack([cx, cy]))
    l1_dist, _ = tree.query(data_xy, p=1)
    print("\nL1 distance (fitted curve vs. all given data points):")
    print(f"  mean   = {l1_dist.mean():.6f}")
    print(f"  median = {np.median(l1_dist):.6f}")
    print(f"  max    = {l1_dist.max():.6f}")
    print(f"  total  = {l1_dist.sum():.6f}  (summed over {len(l1_dist)} points)")

    print("\nDesmos/LaTeX:")
    print(to_latex(theta_deg, M, X))

    plt.figure(figsize=(8, 8))
    plt.scatter(data_xy[:, 0], data_xy[:, 1], s=8, color="tab:blue", alpha=0.6, label="given data")
    plt.plot(cx, cy, color="red", linewidth=1.5, label=f"fitted curve (theta={theta_deg:.2f}, M={M:.4f}, X={X:.2f})")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.title("Fitted curve vs. given data")
    plt.savefig("fit_overlay.png", dpi=150)
    print("\nSaved plot to fit_overlay.png")
