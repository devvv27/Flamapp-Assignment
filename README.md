# Parametric Curve Parameter Estimation

## Problem Statement

The task is to recover three unknown parameters — θ, M, and X — from the parametric curve

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

defined over `6 < t < 60`, with θ constrained to `0°–50°`, M to `−0.05–0.05`, and X to `0–100`. A file, `xy_data.csv`, contains 1500 `(x, y)` points known to lie on this curve.

### Understanding the Geometry

Rearranging the equations to isolate X and 42 makes the underlying structure visible:

```
(x − X) = t·cos(θ) − v·sin(θ)
(y − 42) = t·sin(θ) + v·cos(θ)      where v = e^(M|t|)·sin(0.3t)
```

This is a standard 2D rotation matrix applied to the point `(t, v)`, followed by a translation. In other words, the curve is not an arbitrary shape to be fitted blindly. it is a simple base curve that has been rotated by θ and shifted by X. This observation reframes the problem: recover rotation angle, translation, and growth rate.

### Method 1 — global Chamfer-distance fit

Since the data points carry no `t` label, the natural formulation is a point-cloud-to-curve registration problem rather than a direct equation-solving one:

1. For a candidate `(θ, M, X)`, generate the curve densely over `t ∈ [6, 60]`.
2. For each of the 1500 real data points, find the nearest point on that generated curve (L1 / Manhattan distance, matching the assignment's grading metric), using a `cKDTree` for efficient lookup rather than brute-force comparison.
3. Average these nearest-point distances into a single loss value.
4. Minimize this loss over the full parameter space using `scipy.optimize.differential_evolution`, a population-based, gradient-free global optimizer, followed by a local `Nelder-Mead` polish with a finer curve sampling for precision.

L1 distance was used because its less sensitive to outlier points than a squared (L2) error would be, since it doesn't amplify large individual mismatches the way squaring does. Finding the nearest curve point for all 1500 data points on every single optimizer iteration would be far too slow with brute-force comparison. Therefore a `cKDTree` was used instead, turning each lookup into a fast operation rather than a full scan. For the search itself, `differential_evolution` was chosen over a simple local search because it explores the entire bounded parameter space with a population of candidates rather than improving from one starting guess — important here since the periodic `sin(0.3t)` term makes the loss landscape non-convex, and a purely local method can converge confidently on a wrong answer that only looks like a minimum from nearby. Concretely, the algorithm keeps a pool of candidate `(θ, M, X)` triples, combines and perturbs them across generations and keeps whichever ones score lower on the loss, which lets it hop between distant regions of the search space. A coarser curve sampling (1500 points) was used during this global stage purely for speed, since the objective gets evaluated many thousands of times over the course of the search, while a much finer sampling (20,000 points) was reserved for the final polish, where only a handful of evaluations are needed and precision matters more than speed. That final polish used `Nelder-Mead` rather than a gradient-based method, since the nearest-neighbor lookup makes the loss function technically non-smooth at a fine scale as small parameter changes can abruptly flip which curve point counts as "closest" which confuses methods that rely on a derivative. Nelder-Mead only ever evaluates the function directly, moving and reshaping a small cluster of trial points toward lower loss, so this non-smoothness doesn't affect it the way it would a gradient-based method.

### Method 2: Algebraic Inversion with Closed-Form Regression

This method provides an independent check on the results from the first method. Instead of searching for all unknown values through trial and error, it calculates some of them directly using the structure of the problem, offering a different way to confirm the final answer.


For any guessed pair `(θ, X)`, the rotation can be reversed using direct algebraic calculations:

$$
t = (x - X) \cdot \cos(θ) + (y - 42) \cdot \sin(θ)
$$

$$
v = -(x - X) \cdot \sin(θ) + (y - 42) \cdot \cos(θ)
$$

If `(θ, X)` are correct, the recovered `v` values must follow the true equation:

$$
v = e^{M t} \cdot \sin(0.3t)
$$

Taking the natural logarithm of both sides converts the exponential relationship into a linear equation:

$$
\ln\left(\frac{v}{\sin(0.3t)}\right) = M \cdot t
$$

This allows `M` to be found directly using standard linear regression through the origin—again, with no trial-and-error search.

The reversal for `θ` and `X` is performed using direct algebraic calculations, so no iterative search is required for these two parameters. The logarithm converts the exponential relationship involving `M` into a linear one, which allows `M` to be determined through standard linear regression; this approach has a single direct solution and avoids the instability that can occur with nonlinear searches. Before performing the regression, data points near the zero-crossings of `sin(0.3t)` are excluded, because division by values close to zero amplifies numerical errors into large distortions. Points that yield a negative ratio are also excluded, since the true equation requires a positive value; a negative ratio indicates either an incorrect guess for `(θ, X)` or an unreliable data point. The regression is forced through the origin to match the true model, which has no constant term at `t = 0`; this prevents an intercept from absorbing errors that should be attributed to the parameters instead. Even after `M` is removed from the search, the remaining two-parameter search for `θ` and `X` still contains multiple local solutions due to the periodic nature of the objective function. An initial attempt using the Nelder-Mead optimizer from an arbitrary starting point converged to an incorrect solution (`θ ≈ 14°`, high residual). To avoid these local traps, a coarse grid scan is performed across the entire bounded region to locate the correct area first, followed by local refinement. This scan is computationally inexpensive because each evaluation of the reduced two-parameter objective is fast.

## Final Results

- **θ ≈ 30°** (0.523599 rad)
- **M ≈ 0.03**
- **X ≈ 55**

L1 - Distance:

| Metric | Value |
|---|---|
| Mean | 0.000418 |
| Median | 0.000419 |
| Max | 0.001278 |
| Total (summed) | 0.626523 |

The fitted curve was overlaid on the raw data, both in Python (Matplotlib) and independently in Desmos. In both cases the curve passes through every data point with no visible deviation, including through the most sensitive section of the curve where the wiggle changes direction.

![Fitted curve overlaid on the given data](fit_overlay.png)

*(generated automatically by `fit_curve.py` — saved as `fit_overlay.png` when the script is run)*

## Final Equation

```
\left(t*\cos(0.5236)-e^{0.0300\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.0300\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

Domain: `6 ≤ t ≤ 60`

**Desmos link:** https://www.desmos.com/calculator/hd1qidriwv

## References

- [Breaking Down Nelder-Mead](https://brandewinder.com/2022/03/31/breaking-down-Nelder-Mead/) —Explains how the Nelder-Mead algorithm works
- [Chamfer Distance, explained](https://medium.com/@sim30217/chamfer-distance-4207955e8612) — Explains Chamfer distance metric
- [SciPy `differential_evolution` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) — global optimizer used for Method 1
- [SciPy `minimize` (Nelder-Mead) documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) — local refinement used in both methods
- [SciPy `cKDTree` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) — nearest-point lookup for the Chamfer/L1 distance
- [NumPy](https://numpy.org/) — vectorized curve evaluation and closed-form linear regression
- [pandas](https://pandas.pydata.org/) — CSV loading and data handling
- [Matplotlib](https://matplotlib.org/) — visualization of fitted curve vs. actual data
- [Desmos](https://www.desmos.com/) — independent visual verification of the final equation
