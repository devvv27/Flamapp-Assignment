# Parametric Curve Parameter Estimation

## Problem Statement

The task is to recover three unknown parameters — θ, M, and X — from the parametric curve

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

defined over `6 < t < 60`, with θ constrained to `0°–50°`, M to `−0.05–0.05`, and X to `0–100`. A file, `xy_data.csv`, contains 1500 `(x, y)` points known to lie on this curve — but crucially, without the `t` values that generated them, and not in any particular order. The problem is therefore to recover θ, M, and X purely from the shape of the point cloud.

## Approach

### Step 1: Reading the geometry before touching any numbers

Rearranging the equations to isolate X and 42 makes the underlying structure visible:

```
(x − X) = t·cos(θ) − v·sin(θ)
(y − 42) = t·sin(θ) + v·cos(θ)      where v = e^(M|t|)·sin(0.3t)
```

This is a standard 2D rotation matrix applied to the point `(t, v)`, followed by a translation. In other words, the curve is not an arbitrary shape to be fitted blindly — it is a simple base curve (a straight `t` axis paired with an exponentially-modulated sine wave) that has been rotated by θ and shifted by X. This observation reframed the problem from "fit three abstract parameters" into "recover a rotation angle, a translation, and a growth rate" — a much more concrete target, and one that shaped both methods below.

### Step 2: Method 1 — global Chamfer-distance fit

Since the data points carry no `t` label, the natural formulation is a point-cloud-to-curve registration problem rather than a direct equation-solving one:

1. For a candidate `(θ, M, X)`, generate the curve densely over `t ∈ [6, 60]`.
2. For each of the 1500 real data points, find the nearest point on that generated curve (L1 / Manhattan distance, matching the assignment's grading metric), using a `cKDTree` for efficient lookup rather than brute-force comparison.
3. Average these nearest-point distances into a single loss value.
4. Minimize this loss over the full parameter space using `scipy.optimize.differential_evolution`, a population-based, gradient-free global optimizer, followed by a local `Nelder-Mead` polish with a finer curve sampling for precision.

Each design choice here was made for a specific reason:

- **L1 distance, specifically.** The assignment's own grading criterion is L1 distance, so using it as the fitting loss (rather than, say, Euclidean/L2 distance) keeps the optimization target aligned with how the submission is actually scored. L1 is also less sensitive to any single outlier point than L2, since it doesn't square the error.
- **`cKDTree` instead of brute-force distance comparison.** For every candidate parameter set, the nearest curve point has to be found for all 1500 data points. Brute-force would mean comparing each data point against every one of the (thousands of) sampled curve points — an O(n·m) cost repeated on every optimizer iteration. A KD-tree turns each nearest-neighbor lookup into roughly O(log m), which is what makes running thousands of candidate evaluations during the global search practical at all.
- **`differential_evolution` for the global stage.** This was chosen over a simple multi-restart local search because it doesn't rely on any single starting guess — it maintains and evolves a whole population of candidate solutions across the entire bounded space simultaneously. This matters because the periodic `sin(0.3t)` term makes the loss landscape non-convex: a method that only improves locally from one starting point can converge confidently on a wrong answer that merely looks like a local minimum. A population-based search is far less prone to this.
- **Two different sampling resolutions (1500 points during the global search, 20,000 during polishing).** Differential evolution evaluates the objective function many thousands of times over the course of its search, so a coarser (cheaper) curve sampling was used during that stage purely for speed. Once the search has already landed in the correct region, only a single local refinement is needed, so a much finer sampling can be afforded there to squeeze out precision without materially increasing total runtime.
- **`Nelder-Mead` for the final polish, rather than a gradient-based method.** The objective function passes through a nearest-neighbor lookup, meaning small parameter changes can cause the "closest curve point" to abruptly switch — this makes the loss function technically non-smooth at a fine scale, even though it looks smooth overall. Gradient-based optimizers (e.g. `L-BFGS-B`) assume a differentiable objective and can behave unpredictably here. Nelder-Mead only ever evaluates the function directly and never needs a derivative, so it isn't affected by this.

This converged to θ ≈ 30°, M ≈ 0.03, X ≈ 55, with a mean L1 residual near zero.

### Step 3: Method 2 — algebraic inversion with closed-form regression, as an independent check

Rather than accept Method 1's result on its own, a second, structurally different method was used to verify it. The rotation identified in Step 1 can be undone algebraically for any candidate `(θ, X)`, with no search involved:

```
t = (x − X)·cos(θ) + (y − 42)·sin(θ)
v = −(x − X)·sin(θ) + (y − 42)·cos(θ)
```

If `(θ, X)` are correct, the recovered `v` values should satisfy `v = e^(Mt)·sin(0.3t)` for the true M. Taking a logarithm turns this into a linear equation, `ln(v / sin(0.3t)) = M·t`, which means M can be solved directly by ordinary least-squares regression through the origin — no iterative search required for M at all.

The reasoning behind each step here:

- **Why derive the inversion algebraically instead of continuing to search blindly.** Method 1 treats θ, M, and X as three equally "unknown" quantities to be searched jointly. But once the geometric structure from Step 1 is known, θ and X can be un-done exactly with linear algebra for any guess — there's no reason to spend optimizer iterations discovering something that can be computed directly. This is the whole motivation for Method 2: use the known structure to eliminate as much blind searching as possible.
- **Why the log transform for M, rather than fitting M with another round of optimization.** The relationship `v = e^(Mt)·sin(0.3t)` is nonlinear in M, which would normally call for an iterative solver. But since `t` and `sin(0.3t)` are already known once `(θ,X)` are fixed, taking a logarithm of both sides turns the equation into `ln(v/sin(0.3t)) = M·t` — a straight line through the origin with slope M. A problem that can be reduced to a linear regression is strictly easier and more numerically reliable than one left as a nonlinear search, so this substitution was used wherever possible.
- **Why filter out points where `|sin(0.3t)| ≤ 0.2` before doing the regression.** Near the zero-crossings of `sin(0.3t)`, dividing `v` by a value close to zero amplifies any small numerical noise enormously, which would badly distort the regression. Points close to a zero-crossing were simply excluded from the M calculation, since there are more than enough of the remaining ~1500 points to fit M reliably without them.
- **Why also require the ratio `v/sin(0.3t)` to be positive before taking its log.** Since `e^(M|t|)` is always positive by definition, a negative or near-zero ratio at a given point is a sign that either the candidate `(θ, X)` is wrong (so `v` and `sin(0.3t)` have mismatched signs) or the point is numerically unreliable — either way, including it would inject invalid values into the log calculation, so these points are dropped as well.
- **Why the regression is forced through the origin (no intercept term).** The underlying model has no constant offset — at `t = 0`, the equation requires `ln(ratio) = 0` exactly. Allowing an intercept would let the regression fit a term that has no physical meaning in the original equation, which could quietly absorb some of the fitting error instead of correctly attributing it to M.
- **Why a grid scan still precedes the local refinement, even in this smaller 2-parameter search.** It might seem like reducing the problem to just θ and X should make it easy to optimize directly. In practice it isn't: an initial attempt starting `Nelder-Mead` from an arbitrary point converged to a clearly wrong answer (θ ≈ 14°, high residual), because the `sin(0.3t)` term still creates false local minima even in this reduced space. A coarse grid scan across the full bounded region was therefore run first — cheap here, since each evaluation is now fast — to find the correct basin before refining.

This method converged independently to θ ≈ 30°, M ≈ 0.03, X ≈ 55, matching Method 1 to five-plus decimal places despite using an entirely different mathematical route — one relying on optimization, the other on algebraic inversion plus linear regression.

### Step 4: Validation

**L1 distance.** Using the finalized parameters and a finely sampled version of the fitted curve (50,000 points), the L1 distance from each of the 1500 data points to its nearest curve point was computed directly:

| Metric | Value |
|---|---|
| Mean | 0.000418 |
| Median | 0.000419 |
| Max | 0.001278 |
| Total (summed) | 0.626523 |

Given that the curve spans roughly 50 units in x and 25 in y, distances at the 0.0004-unit scale are effectively noise, not a meaningful mismatch.

**Plotting:** The fitted curve was overlaid on the raw data, both in Python (Matplotlib) and independently in Desmos. In both cases the curve passes through every data point with no visible deviation, including through the most sensitive section of the curve where the wiggle changes direction.

![Fitted curve overlaid on the given data](fit_overlay.png)

*(generated automatically by `fit_curve.py` — saved as `fit_overlay.png` when the script is run)*

## Final Results

- **θ ≈ 30°** (0.523599 rad)
- **M ≈ 0.03**
- **X ≈ 55**

The closeness of these fitted values to round numbers strongly suggests they are the exact parameters used to generate the dataset, with the small deviations attributable to optimizer tolerance rather than any real fitting error.

## Final Equation

```
\left(t*\cos(0.5236)-e^{0.0300\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.0300\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

Domain: `6 ≤ t ≤ 60`

**Desmos link:** https://www.desmos.com/calculator/hd1qidriwv

## Files in This Repo

- `fit_curve.py` — full implementation of both methods, plus L1 distance reporting and plot generation
- `Dataset.csv` — the provided data
- `fit_overlay.png` — fitted curve plotted against the given data
- `README.md` — this file

## Tools & References


- [Breaking Down Nelder-Mead](https://brandewinder.com/2022/03/31/breaking-down-Nelder-Mead/) —Explains how the Nelder-Mead algorithm works
- [Chamfer Distance, explained](https://medium.com/@sim30217/chamfer-distance-4207955e8612) — Explains Chamfer distance metric
- [SciPy `differential_evolution` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) — global optimizer used for Method 1
- [SciPy `minimize` (Nelder-Mead) documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) — local refinement used in both methods
- [SciPy `cKDTree` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) — nearest-point lookup for the Chamfer/L1 distance
- - [NumPy](https://numpy.org/) — vectorized curve evaluation and closed-form linear regression
- [pandas](https://pandas.pydata.org/) — CSV loading and data handling
- [Matplotlib](https://matplotlib.org/) — visualization of fitted curve vs. actual data
- [Desmos](https://www.desmos.com/) — independent visual verification of the final equation
