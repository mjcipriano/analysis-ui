"""Multivariate analyses: principal components, k-means and hierarchical clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import (AnalysisError, COLORS, Context, Results, analysis,
                    new_figure, opt)

CATEGORY = "Multivariate"
GREY = "#9aa3ab"


# ------------------------------------------------------------------ helpers
def _matrix(ctx: Context, min_vars: int = 2, min_rows: int = 3,
            extra: list[str | None] | None = None) -> tuple[pd.DataFrame, list[str], int, np.ndarray]:
    """Complete-case numeric matrix of the chosen variables.

    Returns ``(frame, variable names, rows dropped, index of kept rows)``."""
    names = list(dict.fromkeys(ctx.vars("variables", min=min_vars)))
    if len(names) < min_vars:
        raise AnalysisError(f"Please choose at least {min_vars} different variables.")
    d = pd.DataFrame({c: ctx.numeric(c) for c in names})
    any_data = d.notna().any(axis=1)
    keep = d.notna().all(axis=1).to_numpy()
    dropped = int(any_data.sum() - keep.sum())
    d = d[keep]
    if len(d) < min_rows:
        raise AnalysisError(f"Only {len(d)} row(s) have a value for every chosen variable; at least "
                            f"{min_rows} complete rows are needed.")
    return d, names, dropped, np.where(keep)[0]


def _scale(d: pd.DataFrame, standardize: bool, need_variation: bool = True) -> np.ndarray:
    x = d.to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=1)
    flat = [c for c, s in zip(d.columns, sd) if s == 0]
    if flat and (standardize or need_variation):
        raise AnalysisError("These variables have the same value in every row and cannot be used: "
                            + ", ".join(f"“{c}”" for c in flat) + ".")
    x = x - x.mean(axis=0)
    return x / sd if standardize else x


def _row_labels(ctx: Context, option: str, rows: np.ndarray) -> list[str]:
    name = ctx.var(option, required=False)
    if not name:
        return [str(i + 1) for i in rows]
    col = ctx.column(name).astype(object).to_numpy()[rows]
    out = []
    for i, v in zip(rows, col):
        if pd.isna(v):
            out.append(f"Row {i + 1}")
        elif isinstance(v, float) and v.is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out


def _dropped_warning(r: Results, dropped: int) -> None:
    if dropped:
        r.warn(f"{dropped} row(s) with a missing value in at least one variable were excluded "
               "(complete rows only).")


def pca_fit(x: np.ndarray) -> dict:
    """PCA of an already centred (and possibly scaled) matrix by SVD."""
    n = x.shape[0]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    # deterministic signs: the largest coefficient of every eigenvector is positive
    flip = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
    flip[flip == 0] = 1
    vt, u = vt * flip[:, None], u * flip[None, :]
    eig = s ** 2 / (n - 1)
    return {"eigenvalues": eig, "vectors": vt.T, "scores": u * s,
            "loadings": vt.T * np.sqrt(eig)[None, :]}


def parallel_analysis(n: int, sds: np.ndarray, n_sims: int, percentile: float, seed: int) -> np.ndarray:
    """Percentile of the eigenvalues of random, uncorrelated normal data with
    the same shape (Horn's parallel analysis, Monte Carlo)."""
    rng = np.random.default_rng(seed)
    p = len(sds)
    out = np.empty((n_sims, min(n - 1, p) if n <= p else p))
    for i in range(n_sims):
        z = rng.standard_normal((n, p))
        z -= z.mean(axis=0)
        if np.allclose(sds, 1.0):
            z /= z.std(axis=0, ddof=1)
        else:
            z *= sds
        ev = np.linalg.svd(z, compute_uv=False) ** 2 / (n - 1)
        out[i] = ev[:out.shape[1]]
    return np.percentile(out, percentile, axis=0)


def _color_groups(ctx: Context, rows: np.ndarray) -> tuple[str | None, list[str], np.ndarray]:
    """Group label per kept row for colouring a score plot."""
    name = ctx.var("color", required=False)
    if not name:
        return None, [], np.zeros(len(rows), dtype=int)
    fac = ctx.factor(name)
    levels = [str(c) for c in fac.cat.categories]
    codes = fac.cat.codes.to_numpy()[rows]
    if len(levels) > 12:
        return None, [], np.zeros(len(rows), dtype=int)
    return name, levels, codes


# ======================================================================= PCA
@analysis(
    id="pca", name="Principal component analysis (PCA)", category=CATEGORY,
    description="Reduce many correlated variables to a few principal components.",
    keywords="pca principal components eigenvalues loadings scores scree biplot parallel analysis dimension reduction",
    options=[
        opt.variables("variables", "Variables", opt.CONTINUOUS, min=2),
        opt.variable("color", "Colour points by", opt.ANY, required=False),
        opt.variable("label", "Row labels (optional)", opt.ANY, required=False),
        opt.choice("scaling", "Method", [
            ("standardize", "Standardize (PCA of the correlation matrix)"),
            ("center", "Center only (PCA of the covariance matrix)")], default="standardize"),
        opt.choice("select", "Select components by", [
            ("parallel", "Parallel analysis (Monte Carlo)"), ("kaiser", "Eigenvalue greater than 1 (Kaiser rule)"),
            ("variance", "Percent of total variance explained"), ("fixed", "A fixed number")],
            default="parallel", section="Components"),
        opt.integer("n_sims", "Number of simulations", 1000, min=100, max=5000, section="Components",
                    visible={"select": "parallel"}),
        opt.number("percentile", "Percentile of simulated eigenvalues", 95, min=50, max=99.9, section="Components",
                   visible={"select": "parallel"}),
        opt.integer("seed", "Random seed", 1, min=0, max=10 ** 9, section="Components", visible={"select": "parallel"}),
        opt.number("variance_pct", "Cumulative variance to explain (%)", 80, min=1, max=100, section="Components",
                   visible={"select": "variance"}),
        opt.integer("n_components", "Number of components", 2, min=1, max=100, section="Components",
                    visible={"select": "fixed"}),
        opt.boolean("show_loadings", "Loadings", True, section="Output"),
        opt.boolean("show_vectors", "Eigenvectors", False, section="Output"),
        opt.boolean("show_scores", "PC scores (first 200 rows)", False, section="Output"),
        opt.boolean("show_scree", "Scree plot", True, section="Plots"),
        opt.boolean("show_score_plot", "Score plot (PC1 vs PC2)", True, section="Plots"),
        opt.boolean("show_loading_plot", "Loading plot", True, section="Plots"),
        opt.boolean("show_biplot", "Biplot (scores and loadings)", False, section="Plots"),
    ])
def pca(ctx: Context) -> Results:
    d, names, dropped, rows = _matrix(ctx)
    standardize = ctx.opt("scaling", "standardize") == "standardize"
    x = _scale(d, standardize, need_variation=False)
    n, p = x.shape
    if not standardize and np.allclose(x, 0):
        raise AnalysisError("The variables do not vary, so there is nothing to analyse.")
    fit = pca_fit(x)
    eig = fit["eigenvalues"]
    m = len(eig)
    total = float(eig.sum())
    frac = eig / total
    cum = np.cumsum(frac)

    r = Results()
    _dropped_warning(r, dropped)
    if n <= p:
        r.warn(f"There are only {n} complete rows for {p} variables; at most {n - 1} components carry variance "
               "and the results may be unstable.")

    select = ctx.opt("select", "parallel")
    thresholds = None
    if select == "parallel":
        sds = np.ones(p) if standardize else d.to_numpy(dtype=float).std(axis=0, ddof=1)
        thresholds = parallel_analysis(n, sds, int(ctx.opt("n_sims", 1000)), float(ctx.opt("percentile", 95)),
                                       int(ctx.opt("seed", 1)))
        k = 0
        while k < min(m, len(thresholds)) and eig[k] > thresholds[k]:
            k += 1
        rule = (f"Parallel analysis: components are retained while their eigenvalue exceeds the "
                f"{ctx.opt('percentile', 95):g}th percentile of eigenvalues from {int(ctx.opt('n_sims', 1000))} "
                f"simulated data sets of random normal values with the same number of rows and variables "
                f"(seed {int(ctx.opt('seed', 1))}).")
    elif select == "kaiser":
        cut = 1.0 if standardize else float(eig.sum() / p)
        k = int(np.sum(eig > cut))
        rule = ("Kaiser rule: components with an eigenvalue greater than 1 are retained." if standardize else
                f"Kaiser rule for covariance PCA: components with an eigenvalue above the mean eigenvalue ({cut:.4g}) are retained.")
    elif select == "variance":
        target = float(ctx.opt("variance_pct", 80)) / 100
        k = int(min(m, np.searchsorted(cum, target - 1e-12) + 1))
        rule = f"The smallest number of components explaining at least {100 * target:g}% of the total variance is retained."
    else:
        k = int(min(m, ctx.opt("n_components", 2)))
        rule = f"{k} component(s) retained, as requested."
    k_sel = k
    k_show = max(1, min(m, k))
    pcs = [f"PC{i + 1}" for i in range(m)]

    t = r.table("Eigenvalues", [("pc", "Component", "text"), ("eig", "Eigenvalue", "num"),
                                ("pct", "% of variance", "pct"), ("cum", "Cumulative %", "pct")]
                + ([("thr", "Parallel-analysis threshold", "num")] if thresholds is not None else [])
                + [("sel", "Retained", "text")])
    for i in range(m):
        t.add_row(pc=pcs[i], eig=eig[i], pct=frac[i], cum=cum[i],
                  thr=thresholds[i] if thresholds is not None and i < len(thresholds) else None,
                  sel="Yes" if i < k_sel else "")
    t.note(f"{n} rows, {p} variables. " + ("Variables were standardized (mean 0, SD 1), so the eigenvalues sum to "
                                          f"the number of variables." if standardize else
                                          "Variables were centered but not scaled, so variables with larger variance dominate."))
    t.note(rule)

    if ctx.opt("show_loadings", True):
        t = r.table("Loadings", [("var", "Variable", "text")] + [(f"pc{i}", pcs[i], "num") for i in range(k_show)])
        for j, name in enumerate(names):
            t.add_row(var=name, **{f"pc{i}": fit["loadings"][j, i] for i in range(k_show)})
        t.note("Loading = eigenvector × √eigenvalue"
               + (", the correlation between the variable and the component." if standardize
                  else ", the covariance between the variable and the standardized component."))
        if k_sel == 0:
            t.note("No component passed the selection rule; PC1 is shown for reference.")
    if ctx.opt("show_vectors", False):
        t = r.table("Eigenvectors", [("var", "Variable", "text")] + [(f"pc{i}", pcs[i], "num") for i in range(k_show)])
        for j, name in enumerate(names):
            t.add_row(var=name, **{f"pc{i}": fit["vectors"][j, i] for i in range(k_show)})
        t.note("Coefficients (unit length) that convert the " + ("standardized" if standardize else "centered")
               + " variables into PC scores. Signs are arbitrary; here the largest coefficient of each component is positive.")
    labels = _row_labels(ctx, "label", rows)
    if ctx.opt("show_scores", False):
        t = r.table("PC scores", [("row", "Row", "text")] + [(f"pc{i}", pcs[i], "num") for i in range(k_show)])
        for a in range(min(n, 200)):
            t.add_row(row=labels[a], **{f"pc{i}": fit["scores"][a, i] for i in range(k_show)})
        if n > 200:
            t.note(f"Only the first 200 of {n} rows are shown.")

    lead = (f"PC1 explains {100 * frac[0]:.1f}% of the total variance"
            + (f" and PC2 {100 * frac[1]:.1f}% ({100 * cum[1]:.1f}% together)." if m > 1 else "."))
    how = {"parallel": "Parallel analysis", "kaiser": "The Kaiser rule", "variance": "The variance criterion",
           "fixed": "The requested number"}[select]
    if k_sel == 0:
        sel = f" {how} retained no components: none has a larger eigenvalue than expected for uncorrelated data."
    else:
        sel = (f" {how} retained {k_sel} component{'s' if k_sel != 1 else ''}, "
               f"together explaining {100 * cum[k_sel - 1]:.1f}% of the variance.")
    top = names[int(np.argmax(np.abs(fit["loadings"][:, 0])))]
    r.summary(lead + sel + f" The variable with the largest loading on PC1 is {top}.")

    # ---- plots
    def axis_label(i):
        return f"PC{i + 1} ({100 * frac[i]:.1f}% of variance)"

    if ctx.opt("show_scree", True):
        fig, ax = new_figure(5.2, 3.6)
        xs = np.arange(1, m + 1)
        ax.plot(xs, eig, "-o", color=COLORS[0], ms=5, label="Eigenvalues of the data")
        if thresholds is not None:
            ax.plot(xs[:len(thresholds)], thresholds[:m], "--s", color=COLORS[1], ms=4,
                    label=f"Parallel analysis ({ctx.opt('percentile', 95):g}th percentile)")
        elif select == "kaiser":
            ax.axhline(1.0 if standardize else total / p, color=COLORS[1], ls="--", lw=1, label="Kaiser threshold")
        ax.set_xticks(xs if m <= 20 else xs[::max(1, m // 10)])
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Eigenvalue")
        ax.legend()
        r.plot("Scree plot", fig)

    if m >= 2:
        cname, levels, codes = _color_groups(ctx, rows)
        if ctx.var("color", required=False) and cname is None:
            r.warn("The colour variable has more than 12 groups, so the points are not coloured.")

        def draw_scores(ax, scores, alpha=0.85):
            if cname:
                for gi, lev in enumerate(levels):
                    msk = codes == gi
                    if msk.any():
                        ax.scatter(scores[msk, 0], scores[msk, 1], s=22, color=COLORS[gi % len(COLORS)],
                                   alpha=alpha, label=lev, zorder=3)
                if (codes < 0).any():
                    ax.scatter(scores[codes < 0, 0], scores[codes < 0, 1], s=22, color=GREY, alpha=alpha,
                               label="(missing)", zorder=3)
                ax.legend(title=cname, fontsize=8.5)
            else:
                ax.scatter(scores[:, 0], scores[:, 1], s=22, color=COLORS[0], alpha=alpha, zorder=3)
            ax.axhline(0, color="#cccccc", lw=0.7, zorder=1)
            ax.axvline(0, color="#cccccc", lw=0.7, zorder=1)
            ax.set_xlabel(axis_label(0))
            ax.set_ylabel(axis_label(1))

        def draw_arrows(ax, coords):
            for j, name in enumerate(names):
                ax.annotate("", xy=(coords[j, 0], coords[j, 1]), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=COLORS[1], lw=1.1), zorder=4)
                ax.text(coords[j, 0] * 1.08, coords[j, 1] * 1.08, name, fontsize=8.5, color="#222222",
                        ha="left" if coords[j, 0] >= 0 else "right", va="center", zorder=5)

        if ctx.opt("show_score_plot", True):
            fig, ax = new_figure(5.6, 4.4)
            draw_scores(ax, fit["scores"])
            if ctx.var("label", required=False) and n <= 40:
                for a in range(n):
                    ax.annotate(labels[a], (fit["scores"][a, 0], fit["scores"][a, 1]), xytext=(3, 3),
                                textcoords="offset points", fontsize=7.5, color="#444444")
            r.plot("Score plot", fig)
        if ctx.opt("show_loading_plot", True):
            fig, ax = new_figure(4.8, 4.6)
            L = fit["loadings"][:, :2]
            if standardize:
                th = np.linspace(0, 2 * np.pi, 200)
                ax.plot(np.cos(th), np.sin(th), color="#cccccc", lw=0.8)
                lim = 1.25
            else:
                lim = float(np.abs(L).max()) * 1.35 or 1.0
            ax.axhline(0, color="#cccccc", lw=0.7)
            ax.axvline(0, color="#cccccc", lw=0.7)
            draw_arrows(ax, L)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(axis_label(0))
            ax.set_ylabel(axis_label(1))
            r.plot("Loading plot", fig)
        if ctx.opt("show_biplot", False):
            fig, ax = new_figure(5.8, 4.8)
            sc = fit["scores"][:, :2]
            draw_scores(ax, sc, alpha=0.55)
            V = fit["vectors"][:, :2]
            scale = 0.8 * float(np.abs(sc).max()) / (float(np.abs(V).max()) or 1.0)
            draw_arrows(ax, V * scale)
            r.plot("Biplot", fig)
            r.text("In the biplot, points are PC scores and arrows are the eigenvectors, rescaled to fit the plot.")
    return r


# =================================================================== k-means
def _pc2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = pca_fit(x - x.mean(axis=0))
    frac = f["eigenvalues"] / f["eigenvalues"].sum()
    return f["scores"][:, :2], frac[:2]


@analysis(
    id="kmeans", name="K-means clustering", category=CATEGORY,
    description="Partition rows into k clusters of similar observations.",
    keywords="k-means kmeans cluster clustering silhouette centroids unsupervised",
    options=[
        opt.variables("variables", "Variables", opt.CONTINUOUS, min=2),
        opt.variable("label", "Row labels (optional)", opt.ANY, required=False),
        opt.choice("k_mode", "Number of clusters", [
            ("auto", "Choose automatically (best silhouette, k = 2 to 10)"), ("fixed", "I choose")], default="auto"),
        opt.integer("k", "Clusters (k)", 3, min=2, max=50, visible={"k_mode": "fixed"}),
        opt.boolean("standardize", "Standardize variables", True),
        opt.integer("seed", "Random seed", 1, min=0, max=10 ** 9, section="Algorithm"),
        opt.integer("n_init", "Random starts", 25, min=1, max=200, section="Algorithm"),
        opt.boolean("show_centroids", "Cluster centres", True, section="Output"),
        opt.boolean("show_members", "Cluster membership (first 300 rows)", False, section="Output"),
        opt.boolean("show_plot", "Cluster plot (first two principal components)", True, section="Output"),
        opt.boolean("show_silhouette_plot", "Silhouette by number of clusters", True, section="Output",
                    visible={"k_mode": "auto"}),
    ])
def kmeans(ctx: Context) -> Results:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    d, names, dropped, rows = _matrix(ctx, min_vars=2, min_rows=3)
    standardize = bool(ctx.opt("standardize", True))
    x = _scale(d, standardize, need_variation=False)
    n, p = x.shape
    n_distinct = len(np.unique(x, axis=0))
    if n_distinct < 2:
        raise AnalysisError("All rows are identical, so they cannot be clustered.")
    seed, n_init = int(ctx.opt("seed", 1)), int(ctx.opt("n_init", 25))

    def run(k):
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)

    r = Results()
    _dropped_warning(r, dropped)
    sil_by_k = {}
    if ctx.opt("k_mode", "auto") == "auto":
        kmax = min(10, n_distinct - 1, n - 1)
        if kmax < 2:
            raise AnalysisError("There are too few distinct rows to choose the number of clusters automatically.")
        models = {}
        for k in range(2, kmax + 1):
            models[k] = run(k)
            sil_by_k[k] = float(silhouette_score(x, models[k].labels_))
        k = max(sil_by_k, key=lambda kk: sil_by_k[kk])
        km = models[k]
    else:
        k = int(ctx.opt("k", 3))
        if k > n_distinct:
            raise AnalysisError(f"You asked for {k} clusters, but there are only {n_distinct} distinct complete rows.")
        km = run(k)
    lab = km.labels_
    # number clusters by size (largest = 1) for stable output
    order = np.argsort(-np.bincount(lab, minlength=k), kind="stable")
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    lab = remap[lab]
    centers = km.cluster_centers_[order]
    sil = silhouette_samples(x, lab) if 2 <= k <= n - 1 else np.full(n, np.nan)
    tss = float((x ** 2).sum())
    wss = np.array([float(((x[lab == c] - centers[c]) ** 2).sum()) for c in range(k)])
    bss = tss - float(wss.sum())

    t = r.table("Clusters", [("cluster", "Cluster", "text"), ("n", "Rows", "int"), ("pct", "% of rows", "pct"),
                             ("wss", "Within-cluster SS", "num"), ("sil", "Mean silhouette", "num")])
    for c in range(k):
        t.add_row(cluster=f"Cluster {c + 1}", n=int((lab == c).sum()), pct=float((lab == c).mean()), wss=wss[c],
                  sil=float(np.nanmean(sil[lab == c])))
    t.note(f"K-means (Lloyd's algorithm, k-means++ initialisation, best of {n_init} random starts, seed {seed}) on "
           + ("standardized variables (mean 0, SD 1)." if standardize else "the variables in their original units.")
           + " Clusters are numbered from largest to smallest.")
    t2 = r.table("Sums of squares", [("src", "", "text"), ("ss", "Sum of squares", "num"), ("pct", "% of total", "pct")])
    t2.add_row(src="Between clusters", ss=bss, pct=bss / tss if tss > 0 else None)
    t2.add_row(src="Within clusters", ss=float(wss.sum()), pct=wss.sum() / tss if tss > 0 else None)
    t2.add_row(src="Total", ss=tss, pct=1.0)
    t2.note(f"Average silhouette width = {np.nanmean(sil):.3f} (−1 to 1; values near 1 mean compact, well-separated clusters)."
            + (" Sums of squares are on the standardized scale." if standardize else ""))
    if sil_by_k:
        t3 = r.table("Choice of the number of clusters", [("k", "k", "int"), ("sil", "Average silhouette", "num"),
                                                           ("wss", "Total within-cluster SS", "num"), ("best", "", "text")])
        for kk, s in sil_by_k.items():
            t3.add_row(k=kk, sil=s, wss=float(models[kk].inertia_), best="◄ chosen" if kk == k else "")
        t3.note("The number of clusters with the highest average silhouette width is chosen.")
    if ctx.opt("show_centroids", True):
        orig = d.to_numpy(dtype=float)
        t4 = r.table("Cluster centres (means in original units)",
                     [("var", "Variable", "text")] + [(f"c{c}", f"Cluster {c + 1}", "num") for c in range(k)])
        for j, name in enumerate(names):
            t4.add_row(var=name, **{f"c{c}": float(orig[lab == c, j].mean()) for c in range(k)})
    if ctx.opt("show_members", False):
        labels = _row_labels(ctx, "label", rows)
        t5 = r.table("Cluster membership", [("row", "Row", "text"), ("cluster", "Cluster", "int"), ("sil", "Silhouette", "num")])
        for a in range(min(n, 300)):
            t5.add_row(row=labels[a], cluster=int(lab[a]) + 1, sil=float(sil[a]))
        if n > 300:
            t5.note(f"Only the first 300 of {n} rows are shown.")

    sizes = ", ".join(str(int((lab == c).sum())) for c in range(k))
    r.summary(f"K-means divided the {n} rows into {k} clusters (sizes {sizes})"
              + (", the number with the best average silhouette" if sil_by_k else "")
              + f". The clusters account for {100 * bss / tss:.1f}% of the total variance; average silhouette width "
              f"= {np.nanmean(sil):.2f}. Cluster analysis always finds clusters — check that they make scientific sense.")

    if ctx.opt("show_plot", True):
        sc, frac = _pc2(x)
        fig, ax = new_figure(5.6, 4.4)
        for c in range(k):
            col = COLORS[c % len(COLORS)]
            ax.scatter(sc[lab == c, 0], sc[lab == c, 1], s=22, color=col, alpha=0.85, label=f"Cluster {c + 1}")
            ax.scatter([sc[lab == c, 0].mean()], [sc[lab == c, 1].mean()], marker="X", s=110, color=col,
                       edgecolor="white", linewidth=1.2, zorder=4)
        ax.set_xlabel(f"PC1 ({100 * frac[0]:.1f}% of variance)")
        ax.set_ylabel(f"PC2 ({100 * frac[1]:.1f}% of variance)" if len(frac) > 1 else "PC2")
        ax.legend(fontsize=8.5)
        r.plot("Clusters on the first two principal components", fig)
    if sil_by_k and ctx.opt("show_silhouette_plot", True):
        fig, ax = new_figure(4.8, 3.2)
        ks = list(sil_by_k)
        ax.plot(ks, [sil_by_k[kk] for kk in ks], "-o", color=COLORS[0])
        ax.plot([k], [sil_by_k[k]], "o", ms=10, mfc="none", mec=COLORS[1], mew=1.6)
        ax.set_xticks(ks)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Average silhouette width")
        r.plot("Silhouette by number of clusters", fig)
    return r


# ==================================================== hierarchical clustering
_DIST = {"euclidean": "Euclidean", "manhattan": "Manhattan (city block)", "correlation": "1 − Pearson correlation"}
_LINK = {"ward": "Ward", "average": "Average (UPGMA)", "complete": "Complete (farthest neighbour)",
         "single": "Single (nearest neighbour)"}


@analysis(
    id="hclust", name="Hierarchical clustering", category=CATEGORY,
    description="Build a tree (dendrogram) that groups similar rows or variables.",
    keywords="hierarchical cluster dendrogram ward linkage tree agglomerative heat map",
    options=[
        opt.variables("variables", "Variables", opt.CONTINUOUS, min=2),
        opt.variable("label", "Row labels (optional)", opt.ANY, required=False, visible={"what": "rows"}),
        opt.choice("what", "Cluster", [("rows", "Rows (observations)"), ("columns", "Columns (variables)")],
                   default="rows"),
        opt.choice("distance", "Distance", list(_DIST.items()), default="euclidean", section="Method"),
        opt.choice("linkage", "Linkage", list(_LINK.items()), default="ward", section="Method"),
        opt.boolean("standardize", "Standardize variables", True, section="Method"),
        opt.integer("k", "Cut the tree into this many clusters", None, min=2, max=100, placeholder="none",
                    section="Clusters"),
        opt.boolean("show_members", "Cluster membership (first 300)", True, section="Clusters"),
        opt.boolean("show_plot", "Dendrogram", True, section="Output"),
        opt.boolean("show_steps", "Agglomeration schedule (last 20 merges)", False, section="Output"),
    ])
def hclust(ctx: Context) -> Results:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    what = ctx.opt("what", "rows")
    d, names, dropped, rows = _matrix(ctx, min_vars=2 if what == "rows" else 3, min_rows=3)
    standardize = bool(ctx.opt("standardize", True))
    distance, link = ctx.opt("distance", "euclidean"), ctx.opt("linkage", "ward")
    if link == "ward" and distance != "euclidean":
        raise AnalysisError("Ward's linkage is defined for Euclidean distance only. Choose Euclidean distance "
                            "or a different linkage (for example average).")
    x = _scale(d, standardize, need_variation=(what == "columns"))
    if what == "rows":
        items, labels = x, _row_labels(ctx, "label", rows)
        thing = "rows"
    else:
        items, labels = x.T, names
        thing = "variables"
    m = items.shape[0]
    if m < 3:
        raise AnalysisError(f"At least 3 {thing} are needed to build a tree.")
    if m > 3000:
        raise AnalysisError(f"There are {m} rows; hierarchical clustering here is limited to 3000.")
    if distance == "correlation" and np.any(items.std(axis=1) == 0):
        raise AnalysisError(f"Correlation distance cannot be computed because some {thing} have no variation "
                            + ("across the variables." if what == "rows" else "."))
    dist = pdist(items, metric={"euclidean": "euclidean", "manhattan": "cityblock", "correlation": "correlation"}[distance])
    if not np.all(np.isfinite(dist)):
        raise AnalysisError("Some distances could not be computed; check the data for constant rows or columns.")
    if np.all(dist == 0):
        raise AnalysisError(f"All {thing} are identical, so there is nothing to cluster.")
    Z = hierarchy.linkage(dist, method=link)
    coph = float(hierarchy.cophenet(Z, dist)[0])

    r = Results()
    _dropped_warning(r, dropped)
    t = r.table("Hierarchical clustering", [("item", "", "text"), ("value", "", "text")])
    t.add_row(item=f"Clustered {thing}", value=str(m))
    t.add_row(item="Distance", value=_DIST[distance])
    t.add_row(item="Linkage", value=_LINK[link])
    t.add_row(item="Variables", value=("standardized (mean 0, SD 1)" if standardize else "original units"))
    t.add_row(item="Cophenetic correlation", value=f"{coph:.3f}")
    t.note("Agglomerative clustering. The cophenetic correlation measures how faithfully the tree preserves the "
           "original pairwise distances (closer to 1 is better).")

    k = ctx.opt("k", None)
    threshold = None
    summary = (f"A dendrogram of {m} {thing} was built with {_LINK[link].split(' (')[0].lower()} linkage on "
               f"{_DIST[distance]} distances (cophenetic correlation {coph:.2f}).")
    if k:
        k = int(k)
        if k >= m:
            raise AnalysisError(f"The tree of {m} {thing} cannot be cut into {k} clusters; choose fewer.")
        member = hierarchy.fcluster(Z, t=k, criterion="maxclust")
        found = len(np.unique(member))
        # renumber in dendrogram (left-to-right) order
        leaves = hierarchy.leaves_list(Z)
        seen: dict[int, int] = {}
        for leaf in leaves:
            seen.setdefault(int(member[leaf]), len(seen) + 1)
        member = np.array([seen[int(v)] for v in member])
        if found < k:
            r.warn(f"Because of tied distances the tree could only be cut into {found} clusters, not {k}.")
        if found > 1 and m - found >= 0:
            # half-way between the last merge kept and the first merge cut
            threshold = float((Z[m - found - 1, 2] + Z[m - found, 2]) / 2)
        ts = r.table("Cluster sizes", [("cluster", "Cluster", "text"), ("n", thing.capitalize(), "int")])
        for c in range(1, found + 1):
            ts.add_row(cluster=f"Cluster {c}", n=int((member == c).sum()))
        ts.note("Clusters are numbered in the order they appear in the dendrogram.")
        if ctx.opt("show_members", True):
            tm = r.table("Cluster membership", [("item", "Row" if what == "rows" else "Variable", "text"),
                                                ("cluster", "Cluster", "int")])
            for a in range(min(m, 300)):
                tm.add_row(item=labels[a], cluster=int(member[a]))
            if m > 300:
                tm.note(f"Only the first 300 of {m} are shown.")
        summary += (f" Cutting the tree gave {found} clusters of sizes "
                    + ", ".join(str(int((member == c).sum())) for c in range(1, found + 1)) + ".")
    r.summary(summary)

    if ctx.opt("show_steps", False):
        tz = r.table("Agglomeration schedule", [("step", "Step", "int"), ("a", "Cluster A", "text"), ("b", "Cluster B", "text"),
                                                ("h", "Height (distance)", "num"), ("size", "Size of new cluster", "int")])

        def nm(i):
            i = int(i)
            return labels[i] if i < m else f"(step {i - m + 1})"
        start = max(0, len(Z) - 20)
        for s in range(start, len(Z)):
            tz.add_row(step=s + 1, a=nm(Z[s, 0]), b=nm(Z[s, 1]), h=float(Z[s, 2]), size=int(Z[s, 3]))

    if ctx.opt("show_plot", True):
        named = what == "columns" or bool(ctx.var("label", required=False))
        show_labels = m <= (150 if named else 60)
        horizontal = show_labels and m > 25
        if horizontal:
            fig, ax = new_figure(6.2, max(3.5, 0.12 * m + 1.0))
        else:
            fig, ax = new_figure(max(5.0, 0.32 * m + 1.5) if show_labels else 6.5, 4.0)
        hierarchy.set_link_color_palette(COLORS)
        try:
            hierarchy.dendrogram(Z, ax=ax, labels=labels if show_labels else None, no_labels=not show_labels,
                                 orientation="right" if horizontal else "top",
                                 color_threshold=threshold if threshold is not None else 0,
                                 above_threshold_color="#555555",
                                 leaf_font_size=max(5.5, min(9, 380 / m)) if show_labels else None)
        finally:
            hierarchy.set_link_color_palette(None)
        height_label = "Height (distance)"
        if horizontal:
            ax.set_xlabel(height_label)
            if threshold is not None:
                ax.axvline(threshold, color="#999999", ls="--", lw=0.8)
        else:
            ax.set_ylabel(height_label)
            if threshold is not None:
                ax.axhline(threshold, color="#999999", ls="--", lw=0.8)
            if show_labels and max(len(s) for s in labels) > 3:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(60)
                    lab.set_ha("right")
        r.plot("Dendrogram", fig)
    return r
