"""Nonlinear (and polynomial) curve fitting shared by the curve-fit analyses
and by the fitted-curve overlay of XY graphs.

    fit = fit_model("4pl", x, y)
    fit.params      -> {"Bottom": ..., "Top": ..., "EC50": ..., "HillSlope": ...}
    fit.se, fit.ci  -> standard errors / 95 % CIs per parameter
    fit.r2, fit.aicc, fit.sy_x, fit.dof
    fit.predict(xnew)            -> fitted values
    fit.band(xnew, kind="confidence" | "prediction") -> (lower, upper)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import optimize, stats

from .errors import AnalysisError


@dataclass
class Model:
    id: str
    name: str
    equation: str                       # human readable
    params: list[str]
    func: Callable[..., np.ndarray]     # f(x, *params)
    guess: Callable[[np.ndarray, np.ndarray], Sequence[float]]
    group: str = "Other"
    positive_x: bool = False            # needs x > 0
    lower: Sequence[float] | None = None
    upper: Sequence[float] | None = None
    description: str = ""
    fixed: dict = field(default_factory=dict)   # parameters held constant (see constrain_model)

    @property
    def log_dose(self) -> bool:
        """A dose-response model whose X is a concentration (best shown on a log axis)."""
        return self.group == "Dose–response" and self.positive_x


def _span(y):
    s = float(np.nanmax(y) - np.nanmin(y))
    return s if s > 0 else 1.0


def _xmid(x, y):
    """x where y crosses the half-way point (rough EC50 / half-time guess)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    half = (np.nanmax(ys) + np.nanmin(ys)) / 2
    idx = int(np.argmin(np.abs(ys - half)))
    return float(xs[idx])


def _rate(x):
    r = float(np.nanmax(x) - np.nanmin(x))
    return 1.0 / (r / 3 if r > 0 else 1.0)


def _lin_guess(x, y):
    b, a = np.polyfit(x, y, 1)
    return [a, b]


def _trend(x, y):
    order = np.argsort(x)
    k = max(1, len(x) // 4)
    return float(np.mean(y[order][-k:]) - np.mean(y[order][:k]))


def _g_4pl(x, y):
    hill = 1.0 if _trend(x, y) >= 0 else -1.0
    return [float(np.min(y)), float(np.max(y)), max(_xmid(x, y), 1e-12), hill]


def _g_4pl_log(x, y):
    hill = 1.0 if _trend(x, y) >= 0 else -1.0
    return [float(np.min(y)), float(np.max(y)), _xmid(x, y), hill]


def _g_decay(x, y):
    order = np.argsort(x)
    y0, plateau = float(y[order][0]), float(y[order][-1])
    return [y0, plateau, _rate(x)]


def _g_gauss(x, y):
    i = int(np.argmax(y))
    sd = float(np.std(x)) or 1.0
    return [float(y[i]), float(x[i]), sd / 2]


def _g_power(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() >= 2:
        b, la = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
        return [float(np.exp(la)), float(b)]
    return [1.0, 1.0]


def _g_expgrowth(x, y):
    m = y > 0
    if m.sum() >= 2:
        k, ly0 = np.polyfit(x[m], np.log(y[m]), 1)
        return [float(np.exp(ly0)), float(k)]
    return [float(y[np.argmin(x)]) or 1.0, _rate(x)]


def _g_logistic(x, y):
    y0 = max(float(np.min(y)), 1e-6)
    return [float(np.max(y)) * 1.05, y0, _rate(x) * 3]


def _safe_pow(a, b):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.power(a, b)


MODELS: dict[str, Model] = {}


def _add(m: Model):
    MODELS[m.id] = m


_add(Model("linear", "Straight line", "Y = Intercept + Slope·X", ["Intercept", "Slope"],
           lambda x, a, b: a + b * x, _lin_guess, group="Lines & polynomials"))
_add(Model("quadratic", "Second-order polynomial (quadratic)", "Y = B0 + B1·X + B2·X²",
           ["B0", "B1", "B2"], lambda x, a, b, c: a + b * x + c * x ** 2,
           lambda x, y: list(np.polyfit(x, y, 2)[::-1]), group="Lines & polynomials"))
_add(Model("cubic", "Third-order polynomial (cubic)", "Y = B0 + B1·X + B2·X² + B3·X³",
           ["B0", "B1", "B2", "B3"], lambda x, a, b, c, d: a + b * x + c * x ** 2 + d * x ** 3,
           lambda x, y: list(np.polyfit(x, y, 3)[::-1]), group="Lines & polynomials"))
_add(Model("4pl", "Dose–response: 4-parameter logistic (X = concentration)",
           "Y = Bottom + (Top − Bottom) / (1 + (EC50 / X)^HillSlope)",
           ["Bottom", "Top", "EC50", "HillSlope"],
           lambda x, bottom, top, ec50, hill: bottom + (top - bottom) / (1 + _safe_pow(ec50 / x, hill)),
           _g_4pl, group="Dose–response", positive_x=True,
           lower=[-np.inf, -np.inf, 1e-300, -np.inf],
           description="Sigmoidal dose–response with variable slope. X is the concentration (not its log)."))
_add(Model("3pl", "Dose–response: 3-parameter (Hill slope = 1, X = concentration)",
           "Y = Bottom + (Top − Bottom) / (1 + EC50 / X)", ["Bottom", "Top", "EC50"],
           lambda x, bottom, top, ec50: bottom + (top - bottom) / (1 + ec50 / x),
           lambda x, y: _g_4pl(x, y)[:3], group="Dose–response", positive_x=True,
           lower=[-np.inf, -np.inf, 1e-300]))
_add(Model("5pl", "Dose–response: 5-parameter asymmetric (X = concentration)",
           "Y = Bottom + (Top − Bottom) / (1 + (EC50 / X)^HillSlope)^S",
           ["Bottom", "Top", "EC50", "HillSlope", "S"],
           lambda x, bottom, top, ec50, hill, s: bottom + (top - bottom) / _safe_pow(1 + _safe_pow(ec50 / x, hill), s),
           lambda x, y: _g_4pl(x, y) + [1.0], group="Dose–response", positive_x=True,
           lower=[-np.inf, -np.inf, 1e-300, -np.inf, 1e-3]))
_add(Model("4pl_log", "Dose–response: 4-parameter logistic (X = log10 concentration)",
           "Y = Bottom + (Top − Bottom) / (1 + 10^((LogEC50 − X)·HillSlope))",
           ["Bottom", "Top", "LogEC50", "HillSlope"],
           lambda x, bottom, top, logec50, hill: bottom + (top - bottom) / (1 + _safe_pow(10.0, (logec50 - x) * hill)),
           _g_4pl_log, group="Dose–response"))
_add(Model("michaelis_menten", "Michaelis–Menten", "Y = Vmax·X / (Km + X)", ["Vmax", "Km"],
           lambda x, vmax, km: vmax * x / (km + x),
           lambda x, y: [float(np.max(y)) * 1.1, max(_xmid(x, y), 1e-9)],
           group="Enzyme kinetics & binding"))
_add(Model("hill", "Specific binding with Hill slope", "Y = Bmax·X^h / (Kd^h + X^h)",
           ["Bmax", "h", "Kd"],
           lambda x, bmax, h, kd: bmax * _safe_pow(x, h) / (_safe_pow(kd, h) + _safe_pow(x, h)),
           lambda x, y: [float(np.max(y)) * 1.1, 1.0, max(_xmid(x, y), 1e-9)],
           group="Enzyme kinetics & binding", positive_x=True, lower=[-np.inf, 1e-3, 1e-300]))
_add(Model("one_site_total", "One site: total binding", "Y = Bmax·X / (Kd + X) + NS·X + Background",
           ["Bmax", "Kd", "NS", "Background"],
           lambda x, bmax, kd, ns, bg: bmax * x / (kd + x) + ns * x + bg,
           lambda x, y: [_span(y), max(_xmid(x, y), 1e-9), 0.0, float(np.min(y))],
           group="Enzyme kinetics & binding"))
_add(Model("exp_decay", "One-phase exponential decay", "Y = (Y0 − Plateau)·exp(−K·X) + Plateau",
           ["Y0", "Plateau", "K"],
           lambda x, y0, plateau, k: (y0 - plateau) * np.exp(-k * x) + plateau,
           _g_decay, group="Exponential"))
_add(Model("exp_decay2", "Two-phase exponential decay",
           "Y = Plateau + SpanFast·exp(−KFast·X) + SpanSlow·exp(−KSlow·X)",
           ["Plateau", "SpanFast", "KFast", "SpanSlow", "KSlow"],
           lambda x, plateau, sf, kf, ss, ks: plateau + sf * np.exp(-kf * x) + ss * np.exp(-ks * x),
           lambda x, y: [_g_decay(x, y)[1], (_g_decay(x, y)[0] - _g_decay(x, y)[1]) * 0.6, _rate(x) * 3,
                         (_g_decay(x, y)[0] - _g_decay(x, y)[1]) * 0.4, _rate(x) / 3],
           group="Exponential"))
_add(Model("exp_assoc", "One-phase association", "Y = Y0 + (Plateau − Y0)·(1 − exp(−K·X))",
           ["Y0", "Plateau", "K"],
           lambda x, y0, plateau, k: y0 + (plateau - y0) * (1 - np.exp(-k * x)),
           _g_decay, group="Exponential"))
_add(Model("exp_growth", "Exponential growth", "Y = Y0·exp(k·X)", ["Y0", "k"],
           lambda x, y0, k: y0 * np.exp(k * x), _g_expgrowth, group="Exponential"))
_add(Model("logistic_growth", "Logistic growth", "Y = YM·Y0 / ((YM − Y0)·exp(−k·X) + Y0)",
           ["YM", "Y0", "k"],
           lambda x, ym, y0, k: ym * y0 / ((ym - y0) * np.exp(-k * x) + y0),
           _g_logistic, group="Growth"))
_add(Model("gompertz", "Gompertz growth", "Y = YM·(Y0 / YM)^exp(−K·X)", ["YM", "Y0", "K"],
           lambda x, ym, y0, k: ym * _safe_pow(y0 / ym, np.exp(-k * x)),
           _g_logistic, group="Growth", lower=[1e-300, 1e-300, -np.inf]))
_add(Model("gaussian", "Gaussian", "Y = Amplitude·exp(−0.5·((X − Mean) / SD)²)",
           ["Amplitude", "Mean", "SD"],
           lambda x, amp, mean, sd: amp * np.exp(-0.5 * ((x - mean) / sd) ** 2),
           _g_gauss, group="Other"))
_add(Model("power", "Power series", "Y = A·X^B", ["A", "B"],
           lambda x, a, b: a * _safe_pow(x, b), _g_power, group="Other", positive_x=True))
_add(Model("log_line", "Semi-log line (Y vs ln X)", "Y = A + B·ln(X)", ["A", "B"],
           lambda x, a, b: a + b * np.log(x),
           lambda x, y: list(np.polyfit(np.log(x), y, 1)[::-1]), group="Other", positive_x=True))


def model_choices() -> list[tuple[str, str]]:
    return [(m.id, m.name) for m in MODELS.values()]


def parse_constraints(text: str, model: Model) -> dict[str, float]:
    """Parse ``"Bottom=0, Top=100"`` into ``{"Bottom": 0.0, "Top": 100.0}``.

    Names are matched case-insensitively against the model's parameters."""
    out: dict[str, float] = {}
    lookup = {p.lower(): p for p in model.params}
    for chunk in (c.strip() for c in (text or "").replace(";", ",").replace("\n", ",").split(",")):
        if not chunk:
            continue
        if "=" not in chunk:
            raise AnalysisError(
                f"Could not read the constraint “{chunk}”. Write constraints as Name=value, "
                f"for example “{model.params[0]}=0”.")
        name, _, value = chunk.partition("=")
        key = lookup.get(name.strip().lower())
        if key is None:
            raise AnalysisError(
                f"“{name.strip()}” is not a parameter of “{model.name}”. "
                f"Its parameters are: {', '.join(model.params)}.")
        try:
            val = float(value.strip().replace("−", "-"))
        except ValueError:
            raise AnalysisError(f"The constraint “{chunk}” needs a number after the equals sign.") from None
        if not np.isfinite(val):
            raise AnalysisError(f"The constraint “{chunk}” needs a finite number.")
        out[key] = val
    return out


def constrain_model(model: Model, fixed: dict[str, float] | None) -> Model:
    """A copy of ``model`` in which the parameters in ``fixed`` are constants."""
    if not fixed:
        return model
    unknown = [k for k in fixed if k not in model.params]
    if unknown:
        raise AnalysisError(f"Unknown parameter(s) for “{model.name}”: {', '.join(unknown)}.")
    free_idx = [i for i, p in enumerate(model.params) if p not in fixed]
    if not free_idx:
        raise AnalysisError("Every parameter is fixed, so there is nothing left to fit.")
    template = [fixed.get(p, np.nan) for p in model.params]
    for i, p in enumerate(model.params):
        if p in fixed:
            lo = model.lower[i] if model.lower is not None else -np.inf
            hi = model.upper[i] if model.upper is not None else np.inf
            if not (lo <= fixed[p] <= hi):
                raise AnalysisError(f"{p} cannot be fixed to {fixed[p]:g} in “{model.name}” (outside its allowed range).")

    def func(x, *free):
        full = list(template)
        for i, v in zip(free_idx, free):
            full[i] = v
        return model.func(x, *full)

    def guess(x, y):
        g = list(model.guess(x, y))
        return [g[i] for i in free_idx]

    eq = model.equation + "   [fixed: " + ", ".join(f"{k} = {v:g}" for k, v in fixed.items()) + "]"
    return Model(id=model.id, name=model.name, equation=eq,
                 params=[model.params[i] for i in free_idx], func=func, guess=guess,
                 group=model.group, positive_x=model.positive_x,
                 lower=None if model.lower is None else [model.lower[i] for i in free_idx],
                 upper=None if model.upper is None else [model.upper[i] for i in free_idx],
                 description=model.description, fixed=dict(fixed))


@dataclass
class Fit:
    model: Model
    x: np.ndarray
    y: np.ndarray
    popt: np.ndarray
    pcov: np.ndarray
    weights: np.ndarray | None = None
    converged: bool = True
    weighting: str = "none"
    params: dict = field(default_factory=dict)
    se: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)

    def __post_init__(self):
        n, p = len(self.x), len(self.popt)
        self.n, self.k = n, p
        self.dof = n - p
        self.fitted = self.model.func(self.x, *self.popt)
        self.residuals = self.y - self.fitted
        w = self.weights if self.weights is not None else np.ones(n)
        self.ss_res = float(np.sum(w * self.residuals ** 2))
        ybar = np.average(self.y, weights=w)
        self.ss_tot = float(np.sum(w * (self.y - ybar) ** 2))
        self.r2 = 1 - self.ss_res / self.ss_tot if self.ss_tot > 0 else float("nan")
        self.adj_r2 = (1 - (1 - self.r2) * (n - 1) / self.dof) if self.dof > 0 else float("nan")
        self.sy_x = float(np.sqrt(self.ss_res / self.dof)) if self.dof > 0 else float("nan")
        if self.ss_res > 0 and n > 0:
            kk = p + 1
            self.aic = n * np.log(self.ss_res / n) + 2 * kk
            self.aicc = self.aic + (2 * kk * (kk + 1)) / (n - kk - 1) if n - kk - 1 > 0 else float("nan")
        else:
            self.aic = self.aicc = float("nan")
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.diag(self.pcov))
        tcrit = stats.t.ppf(0.975, self.dof) if self.dof > 0 else float("nan")
        for name, val, s in zip(self.model.params, self.popt, se):
            self.params[name] = float(val)
            self.se[name] = float(s)
            self.ci[name] = (float(val - tcrit * s), float(val + tcrit * s))

    def predict(self, x) -> np.ndarray:
        return self.model.func(np.asarray(x, dtype=float), *self.popt)

    def _jac(self, x: np.ndarray) -> np.ndarray:
        p = self.popt
        J = np.empty((len(x), len(p)))
        for i in range(len(p)):
            h = 1e-6 * max(abs(p[i]), 1e-6)
            up, dn = p.copy(), p.copy()
            up[i] += h
            dn[i] -= h
            J[:, i] = (self.model.func(x, *up) - self.model.func(x, *dn)) / (2 * h)
        return J

    def band(self, x, kind: str = "confidence", level: float = 0.95):
        """Delta-method confidence (or prediction) band at ``x``."""
        x = np.asarray(x, dtype=float)
        yhat = self.predict(x)
        if self.dof <= 0 or not np.all(np.isfinite(self.pcov)):
            nan = np.full_like(yhat, np.nan)
            return nan, nan
        J = self._jac(x)
        var = np.einsum("ij,jk,ik->i", J, self.pcov, J)
        if kind == "prediction":
            # scatter of one new observation; with weighting its variance is Sy.x² / weight
            if self.weighting in ("1/y", "1/y_obs"):
                scale = np.abs(yhat)
            elif self.weighting in ("1/y2", "1/y2_obs"):
                scale = yhat ** 2
            elif self.weights is not None:
                scale = float(np.mean(1 / self.weights))
            else:
                scale = 1.0
            var = var + self.sy_x ** 2 * scale
        half = stats.t.ppf(0.5 + level / 2, self.dof) * np.sqrt(np.maximum(var, 0))
        return yhat - half, yhat + half

    # ------------------------------------------------ profile likelihood
    def _profile_ss(self, index: int, value: float, start: np.ndarray,
                    thorough: bool = False) -> tuple[float, np.ndarray]:
        """Weighted SS with parameter ``index`` held at ``value`` and all
        other parameters re-optimised (starting from ``start``; with
        ``thorough`` also from the model's own guess and perturbed starts, to
        escape a local minimum)."""
        names = self.model.params
        sigma = None if self.weights is None else 1 / np.sqrt(self.weights)
        w = self.weights if self.weights is not None else 1.0
        if len(names) == 1:
            pred = self.model.func(self.x, value)
            return float(np.sum(w * (self.y - pred) ** 2)), np.empty(0)
        cm = constrain_model(self.model, {names[index]: float(value)})
        starts = [np.asarray(start, dtype=float)]
        maxfev = 300 * len(names)
        if thorough:
            try:
                starts.append(np.asarray([float(v) for v in cm.guess(self.x, self.y)]))
            except Exception:
                pass
            rng = np.random.default_rng(1)
            starts += [starts[0] * rng.uniform(0.5, 2.0, len(starts[0])) for _ in range(2)]
        best: tuple[float, np.ndarray] | None = None
        for s0 in starts:
            self._profile_budget -= 1
            if self._profile_budget < 0:
                raise RuntimeError("profile search budget exhausted")
            try:
                others, _ = _lsq(cm, self.x, self.y, sigma, s0, maxfev)
                ss = float(np.sum(w * (self.y - cm.func(self.x, *others)) ** 2))
            except Exception:
                continue
            if np.isfinite(ss) and (best is None or ss < best[0]):
                best = (ss, others)
        if best is None:
            raise RuntimeError("profile refit failed")
        return best

    def profile_ci(self, level: float = 0.95) -> dict:
        """Asymmetric profile-likelihood confidence intervals, as Prism
        reports by default (and R's ``confint(nls)``).

        For each parameter the limit is the value at which the sum of squares,
        re-minimised over all *other* parameters, has risen to
        ``SS_min · (1 + F(level; 1, dof) / dof)``. A limit that cannot be
        found (the SS never rises that far, the parameter runs into a bound of
        the model, the refit fails, or the search uses up its budget of
        refits) is NaN — "very wide" in Prism's terms.
        With weighting, the weights of the best fit are kept fixed."""
        cache = self.__dict__.setdefault("_profile_cache", {})
        if level in cache:
            return cache[level]
        names = self.model.params
        out = {n: (float("nan"), float("nan")) for n in names}
        if self.dof > 0 and self.ss_res > 0 and np.all(np.isfinite(self.popt)):
            target = self.ss_res * (1 + stats.f.ppf(level, 1, self.dof) / self.dof)
            bounds = _bounds(self.model)
            for i, name in enumerate(names):
                lims = []
                for sign in (-1.0, 1.0):
                    self._profile_budget = 70     # refits allowed per limit; beyond that the limit is "very wide"
                    bound = None if bounds is None else bounds[0 if sign < 0 else 1][i]
                    try:
                        with np.errstate(all="ignore"):
                            lims.append(self._profile_limit(i, sign, target, bound))
                    except Exception:
                        lims.append(float("nan"))
                out[name] = (lims[0], lims[1])
        cache[level] = out
        return out

    def _profile_limit(self, i: int, sign: float, target: float, bound: float | None) -> float:
        best = float(self.popt[i])
        se = self.se[self.model.params[i]]
        step = se if np.isfinite(se) and se > 0 else max(abs(best) * 0.1, 1e-3)
        prev_v, prev_p = best, np.delete(self.popt, i)
        delta = step * 0.5
        for _ in range(40):
            v = prev_v + sign * delta
            at_bound = False
            if bound is not None and np.isfinite(bound) and (v - bound) * sign >= 0:
                v, at_bound = float(bound), True
            ss, p = self._profile_ss(i, v, prev_p)
            if ss >= target:        # make sure this is not a stuck refit before trusting it
                ss, p = self._profile_ss(i, v, prev_p, thorough=True)
            if ss >= target:
                state = {"p": prev_p}

                def g(val):
                    s_, p_ = self._profile_ss(i, val, state["p"])
                    return s_ - target
                root = float(optimize.brentq(g, prev_v, v, xtol=1e-7 * max(abs(best), step), rtol=1e-9, maxiter=60))
                ss_r, p_r = self._profile_ss(i, root, prev_p, thorough=True)
                if ss_r >= target * (1 - 1e-4):
                    return root
                prev_v, prev_p = root, p_r      # the crossing was an artefact of a local minimum: keep going
                continue
            if at_bound or abs(v - best) > 1e6 * step:
                return float("nan")
            prev_v, prev_p = v, p
            delta *= 2.0
        return float("nan")

    def equation_text(self, digits: int = 4) -> str:
        """The fitted equation with numbers, e.g. for graph legends."""
        parts = [f"{k} = {v:.{digits}g}" for k, v in self.params.items()]
        return ", ".join(parts)

    def solve_x(self, y_target: float) -> float:
        """Interpolate: the x within (slightly beyond) the data range where the
        curve equals ``y_target``; NaN when there is no crossing."""
        lo, hi = float(np.min(self.x)), float(np.max(self.x))
        pad = (hi - lo) * 0.5
        lo2 = lo - pad if not self.model.positive_x else max(lo / 10, 1e-300)
        hi2 = hi + pad if not self.model.positive_x else hi * 10
        if self.model.positive_x:
            grid = np.geomspace(lo2, hi2, 2000)
        else:
            grid = np.linspace(lo2, hi2, 2000)
        with np.errstate(all="ignore"):
            f = self.predict(grid) - y_target
        ok = np.isfinite(f)
        grid, f = grid[ok], f[ok]
        sign = np.where(np.diff(np.sign(f)) != 0)[0]
        if len(sign) == 0:
            return float("nan")
        # prefer a crossing inside the observed x range
        best = min(sign, key=lambda i: 0 if lo <= grid[i] <= hi else 1)
        try:
            return float(optimize.brentq(lambda v: float(self.predict([v])[0]) - y_target,
                                         grid[best], grid[best + 1]))
        except ValueError:
            return float("nan")


def fit_model(model_id: str | Model, x, y, weighting: str = "none", maxfev: int = 20000,
              fixed: dict[str, float] | None = None) -> Fit:
    """Least-squares fit of a built-in model.

    ``weighting`` (see ``WEIGHTINGS``): "none"; "1/y" (Poisson) and "1/y2"
    (relative) weight by the *fitted* Y through iteratively reweighted least
    squares, as Prism does; "1/y_obs" / "1/y2_obs" use the observed Y (as
    ``nls(weights = 1/y^2)`` in R); "1/sd2" uses the SD of the replicates at
    each X.
    ``fixed``: parameters to hold constant, e.g. ``{"Bottom": 0, "Top": 100}``.
    ``model_id`` may also be a :class:`Model` instance.
    """
    if isinstance(model_id, Model):
        m = model_id
    elif model_id not in MODELS:
        raise AnalysisError(f"Unknown model: {model_id}")
    else:
        m = MODELS[model_id]
    m = constrain_model(m, fixed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if m.positive_x:
        keep &= x > 0
    x, y = x[keep], y[keep]
    if len(x) < len(m.params) + 1:
        need = len(m.params) + 1
        extra = " with X > 0" if m.positive_x else ""
        raise AnalysisError(
            f"“{m.name}” has {len(m.params)} parameters and needs at least {need} points{extra}; "
            f"only {len(x)} are available.")
    if np.ptp(x) == 0:
        raise AnalysisError("All X values are identical, so a curve cannot be fitted.")
    sigma = _initial_sigma(weighting, x, y)
    try:
        p0 = [float(v) for v in m.guess(x, y)]
    except Exception:
        p0 = [1.0] * len(m.params)
    p0 = [v if np.isfinite(v) else 1.0 for v in p0]
    bounds = _bounds(m)
    starts = [p0]
    rng = np.random.default_rng(0)
    for _ in range(6):
        starts.append([v * float(rng.uniform(0.3, 3.0)) if v != 0 else float(rng.normal(0, 1)) for v in p0])
    best = None
    last_err: Exception | None = None
    for i_start, s in enumerate(starts):
        try:
            popt, pcov = _lsq(m, x, y, sigma, s, maxfev)
            res = y - m.func(x, *popt)
            ss = float(np.sum((res / (sigma if sigma is not None else 1.0)) ** 2))
            if not np.isfinite(ss):
                continue
            if best is None or ss < best[0] * (1 - 1e-9):
                best = (ss, popt, pcov)
            if best is not None and i_start == 0 and np.all(np.isfinite(pcov)):
                break   # the informed start converged cleanly; good enough
        except Exception as exc:  # RuntimeError (no convergence) / ValueError
            last_err = exc
    if best is None:
        raise AnalysisError(
            f"The fit of “{m.name}” did not converge. The model may not describe these data; "
            f"try a different model. ({last_err})")
    _, popt, pcov = best
    if weighting in ("1/y", "1/y2"):
        # Iteratively reweighted least squares: weights from the *fitted* Y,
        # recomputed after every fit until the parameters stop changing.
        converged = False
        for _ in range(100):
            sigma = _sigma_from(weighting, m.func(x, *popt), y)
            try:
                new, pcov = _lsq(m, x, y, sigma, popt, maxfev)
            except Exception as exc:
                raise AnalysisError(
                    f"The weighted fit of “{m.name}” did not converge ({exc}). "
                    "Try no weighting or a different model.") from None
            change = np.max(np.abs(new - popt) / (np.abs(popt) + 1e-12))
            popt = new
            if change < 1e-9:
                converged = True
                break
        if not converged or not np.all(np.isfinite(popt)):
            raise AnalysisError(
                f"The weighted fit of “{m.name}” did not settle on stable weights. "
                "Try no weighting or a different model.")
        sigma = _sigma_from(weighting, m.func(x, *popt), y)
    weights = None if sigma is None else 1 / sigma ** 2
    return Fit(model=m, x=x, y=y, popt=np.asarray(popt, dtype=float),
               pcov=np.asarray(pcov, dtype=float), weights=weights, weighting=weighting)


WEIGHTINGS = {
    "none": "No weighting",
    "1/y": "1/Y (Poisson; weights from the fitted Y)",
    "1/y2": "1/Y² (relative; weights from the fitted Y)",
    "1/y_obs": "1/Y (weights from the observed Y)",
    "1/y2_obs": "1/Y² (weights from the observed Y)",
    "1/sd2": "1/SD² of the replicates at each X",
}


def _sigma_from(weighting: str, basis: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard-deviation scale per point for 1/Y or 1/Y² weighting, given
    the Y values (observed or fitted) the weights are based on."""
    a = np.abs(np.asarray(basis, dtype=float))
    scale = float(np.max(np.abs(y))) if len(y) else 1.0
    floor = max(scale * 1e-6, 1e-300)
    a = np.where(np.isfinite(a) & (a > floor), a, floor)
    return np.sqrt(a) if weighting.startswith("1/y") and "2" not in weighting else a


def _initial_sigma(weighting: str, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if weighting in (None, "", "none"):
        return None
    if weighting in ("1/y", "1/y2", "1/y_obs", "1/y2_obs"):
        ay = np.abs(y)
        ay = np.where(ay == 0, np.min(ay[ay > 0]) if np.any(ay > 0) else 1.0, ay)
        return _sigma_from(weighting, ay, y)
    if weighting == "1/sd2":
        sd = np.empty(len(x))
        for ux in np.unique(x):
            sel = x == ux
            if sel.sum() < 2:
                raise AnalysisError(
                    "Weighting by 1/SD² needs at least two replicate Y values at every X, "
                    f"but X = {ux:g} has only one.")
            sd[sel] = np.std(y[sel], ddof=1)
        if np.any(sd <= 0):
            raise AnalysisError(
                "Weighting by 1/SD² is impossible because the replicates at some X values are identical (SD = 0).")
        return sd
    raise AnalysisError(f"Unknown weighting: {weighting}")


def _bounds(m: Model):
    if m.lower is None and m.upper is None:
        return None
    lo = list(m.lower) if m.lower is not None else [-np.inf] * len(m.params)
    hi = list(m.upper) if m.upper is not None else [np.inf] * len(m.params)
    return lo, hi


def _lsq(m: Model, x, y, sigma, p0, maxfev: int = 20000):
    """One (weighted) least-squares fit from ``p0``; raises when it fails."""
    bounds = _bounds(m)
    kw = dict(p0=list(p0), sigma=sigma, absolute_sigma=False, maxfev=maxfev)
    if bounds is not None:
        kw["p0"] = [min(max(v, l + (1e-12 if np.isfinite(l) else 0)), h)
                    for v, l, h in zip(p0, bounds[0], bounds[1])]
        kw["bounds"] = bounds
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = optimize.curve_fit(m.func, x, y, **kw)
    return np.asarray(popt, dtype=float), np.asarray(pcov, dtype=float)


def compare_fits(simple: Fit, complex_: Fit) -> dict:
    """Extra-sum-of-squares F test between two nested fits of the same data."""
    df1 = simple.dof - complex_.dof
    if df1 <= 0 or complex_.dof <= 0:
        return {"F": float("nan"), "df1": df1, "df2": complex_.dof, "p": float("nan")}
    F = ((simple.ss_res - complex_.ss_res) / df1) / (complex_.ss_res / complex_.dof)
    p = float(stats.f.sf(F, df1, complex_.dof)) if F > 0 else 1.0
    return {"F": float(F), "df1": int(df1), "df2": int(complex_.dof), "p": p}
