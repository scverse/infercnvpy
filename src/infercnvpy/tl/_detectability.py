"""Which chromosome-arm copy number changes a gene panel and sequencing depth can detect (`tl.detectability`)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import norm

from infercnvpy._logging import SKIP_OWN_FRAMES, log_result
from infercnvpy._numba import n_kernel_chunks, numba_threads
from infercnvpy._util import _as_df, _as_matrix, _obs_column, _params_for_uns
from infercnvpy.tl._detectability_kernels import _arm_information, _nb_moments
from infercnvpy.tl._infercnv import _categories, _natural_sort

#: GRCh38 centromere midpoints (Mb), from the "acen" bands of the UCSC cytoband table
_CENTROMERES_MB = {
    "chr1": 123.4, "chr2": 93.9, "chr3": 90.9, "chr4": 50.0, "chr5": 48.8, "chr6": 59.8, "chr7": 60.1, "chr8": 45.2,
    "chr9": 43.0, "chr10": 39.8, "chr11": 53.4, "chr12": 35.5, "chr13": 17.7, "chr14": 17.2, "chr15": 19.0,
    "chr16": 36.8, "chr17": 25.1, "chr18": 18.5, "chr19": 26.2, "chr20": 28.1, "chr21": 12.0, "chr22": 15.0,
    "chrX": 60.6, "chrY": 10.4,
}  # fmt: skip
HG38_CENTROMERES = {k: v * 1e6 for k, v in _CENTROMERES_MB.items()}
#: p arms of the acrocentric chromosomes, which hold no genes of their own: listed only when the panel has genes there
_ACROCENTRIC_P = {"chr13p", "chr14p", "chr15p", "chr21p", "chr22p"}
_MAX_PHI = 100.0  # cap of the NB dispersion 1 / k of a gene
_MAX_GAIN = 1000.0  # largest copy ratio searched for `gain_ratio`


def detectability(
    data: AnnData | Sequence[str],
    *,
    expression: AnnData | None = None,
    layer: str | None = None,
    groupby: str | None = None,
    reference_key: str | None = None,
    reference_cat: str | Sequence[str] | None = None,
    counts_per_cell: float | Sequence[float] | None = None,
    n_cells: int = 100,
    gain: float = 1.5,
    loss: float = 0.5,
    alpha: float = 1e-3,
    power: float = 0.8,
    centromeres: Mapping[str, float] | None = None,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    random_state: int = 0,
    n_jobs: int | None = None,
    key_added: str = "detectability",
    inplace: bool = True,
) -> pd.DataFrame | None:
    """Which copy number changes of each chromosome arm a panel and depth can detect, and in how many cells.

    For every chromosome arm, this reports the smallest gain and loss that a test on `n_cells` cells detects, and the
    number of cells it needs to detect a `gain` (default 1.5x) and a `loss` (default 0.5x), from the expression and
    depth of the data. Use it before an experiment, with a gene list and expression from a similar dataset, to choose
    a panel and depth, or after it, on the data itself, to see which arms its CNV calls can be trusted on.

    Two numbers are reported for each: an ideal bound from counting noise alone, and the headline "achievable" value,
    which also includes the extra variance that diploid cells show on the arm. On small marker panels, cell type and
    cell state programs move the few genes of an arm together, which looks like copy number. The ideal bound ignores
    that and over-promises; the gap between the two columns shows by how much.

    **Model.** A diploid cell expects `m = s p` counts of a gene, with `p` the gene's share of the counts of the cell's
    group (`groupby`) and `s` the cell's size: its counts off the arm, divided by the share of the genes off the arm,
    so that the arm's own counts don't enter. Counts are negative binomial with mean `m` and dispersion `phi = 1 / k`
    (size `k`), fitted per gene by the method of moments on the reference cells. A copy ratio `r` of the arm
    multiplies `m` by `r`. The cell's efficient score for the arm, `U = sum (x - m) / (1 + m phi)` over the arm's genes,
    has mean 0 and variance `I = sum m / (1 + m phi)` without a change. `I` is the Fisher information about the log
    copy ratio of the negative binomial with a log link :cite:`Lawless1987`, the working weight that DESeq2 uses for
    its Wald test :cite:`Love2014`. With a change, `U` has mean `(r - 1) I` and variance `r A + r^2 B` (`A` and `B`
    are the parts of `I` from counting and from dispersion).

    **Null variance.** Genes of an arm co-vary within cells (cell type and state programs), which the model misses.
    `null_var` measures it on the reference cells as `sum U^2 / sum I` per arm. Counting noise alone gives 1, and the
    noise of each cell's size factor adds a little (1.1 to 1.3 in simulations). The achievable values divide the
    information by it (at least 1); the ideal values use 1. Give the cell types as `groupby`, so that differences
    between types are part of the expectation and not of the null variance.

    **Test.** A one-sided z-test of the score summed over `n` cells, at level `alpha` and with `power`
    :cite:`Cox1974`. With `I`, `A` and `B` averaged over the cells of a group and `v` the null variance, it detects
    ratio `r` when `|r - 1| sqrt(n) I >= sqrt(v) (z_alpha sqrt(I) + z_power sqrt(r A + r^2 B))`. Solved for `n`, this
    gives the cells needed; solved for `r`, the smallest detectable ratio. The test assumes that cells are
    independent: a program shared by all cells of a clone (e.g. a tumour state) is not averaged away by more cells.
    It is the bound for a method that knows the diploid expression of every cell type. Methods that estimate it from
    the data reach less: in our simulations on imaging panels, :func:`infercnvpy.tl.infercnv` reached about 0.6 of
    the z of this test, so it needed about 3 times the cells. That factor is not yet calibrated per platform.

    **Depths.** With `counts_per_cell`, the counts are thinned at random to each target median per cell, and
    everything is measured again. A target above the observed depth scales the expected counts up instead, and
    extrapolates the excess of the null variance in proportion to the information (`null_var_source="extrapolated"`).

    **Gene-list mode.** Pass a list of gene symbols as `data` and an AnnData of raw counts from a similar tissue as
    `expression`. The panel is then cut out of `expression`, so the result describes `expression`'s cells at
    `expression`'s depth on the panel (set the depth of your platform with `counts_per_cell`; imaging panels often
    have more counts per gene than a droplet dataset). A bundled atlas table, so that no expression data is needed,
    is planned.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    data
        Raw counts as AnnData (cells x genes, with gene positions in `var["chromosome"]` and `var["start"]`), or a
        list of gene symbols (gene-list mode, needs `expression`).
    expression
        Gene-list mode only: AnnData of raw counts and gene positions that gives the expression of the panel genes.
        `layer`, `groupby` and the reference refer to it.
    layer
        Layer with the raw counts. Default: `X`.
    groupby
        Column of `obs` with cell types. Each type gets its own expected expression and its own rows in the result.
        Default: all cells are one group, `"all"`.
    reference_key
        Column of `obs` that marks the diploid reference cells.
    reference_cat
        One or more categories of `reference_key` that are diploid. The NB dispersion and the null variance are
        measured on these cells. Default: all cells, which counts real copy number changes as null variance (the
        result is then conservative).
    counts_per_cell
        Target median counts per cell, one value or a list (one set of rows per value). Default: the observed depth.
    n_cells
        Number of cells for `gain_ratio` and `loss_ratio`. `1` gives what a single cell shows.
    gain
        Copy ratio of the gain for `cells_gain` (1.5 is one extra copy of a diploid arm).
    loss
        Copy ratio of the loss for `cells_loss` (0.5 is one copy lost).
    alpha
        One-sided level of the test of each arm. The default, 0.001, is about 0.05 with a Bonferroni correction for 40
        arms.
    power
        Power of the test.
    centromeres
        Centromere position (bp) of each chromosome, which splits it into the p and q arm. Default: GRCh38.
    exclude_chromosomes
        Chromosomes to leave out.
    random_state
        Seed of the thinning for `counts_per_cell`.
    n_jobs
        Number of numba threads. Default: numba's setting.
    key_added
        Key of the result in `adata.uns`, and of the per-cell information in `adata.obs`.
    inplace
        If `True`, store the result in adata, otherwise return it. In gene-list mode, the result is always returned.

    Returns
    -------
    If `inplace=True` and `data` is an AnnData, returns `None` and sets the fields below. Otherwise returns the table
    of `adata.uns[key_added]["table"]`.

    `adata.uns[key_added]["table"]` : :class:`pandas.DataFrame`
        One row per group, depth and arm (in genome order), with the columns

        * `group`: the cell type (`"all"` without `groupby`);
        * `depth`: the median counts per cell of all cells (the target of `counts_per_cell`);
        * `counts_per_cell`: the median counts per cell of the group;
        * `arm`, `chromosome`: the arm (e.g. `"chr1p"`) and its chromosome;
        * `n_genes`: panel genes on the arm. Arms with fewer than about 5 genes give fragile results;
        * `info`: Fisher information per cell about the log copy ratio of the arm (mean over the group's cells);
        * `null_var`: variance of the arm's score in reference cells relative to the model (1: counting noise only);
        * `null_var_source`: `"measured"` (on the counts, thinned to the depth) or `"extrapolated"` (a depth above the
          data's);
        * `gain_ratio`, `loss_ratio`: smallest gain and mildest loss (copy ratios) detectable in `n_cells` cells, NaN
          if no gain up to 1000x, or no loss down to 0, is;
        * `cells_gain`, `cells_loss`: cells needed to detect a `gain` and a `loss` (`inf` without information);
        * the same four with the suffix `_ideal`: the counting bound, without the null variance.

        Rows are sorted by group, depth and arm. Values with the suffix are optimistic; use the plain ones.
    `adata.uns[key_added]["params"]` : :class:`dict`
        Parameters of this call, and `n_reference` (number of reference cells).
    `adata.obs[f"{key_added}_info"]` : float
        Per-cell information summed over arms at the observed depth, a per-cell measure of how much a cell can say
        about copy number.

    See Also
    --------
    infercnvpy.pl.detectability : plot the result.

    Examples
    --------
    >>> import infercnvpy as cnv
    >>> cnv.tl.detectability(adata, groupby="cell_type", reference_key="cell_type", reference_cat=["T cell", "B cell"])
    >>> cnv.pl.detectability(adata, metric="cells_gain")

    Before an experiment, for a panel at 200 counts per cell, with expression from a droplet atlas of the tissue:

    >>> cnv.tl.detectability(panel_genes, expression=atlas, groupby="cell_type", counts_per_cell=[100, 200])
    """
    _check_numbers(n_cells=n_cells, gain=gain, loss=loss, alpha=alpha, power=power)
    if isinstance(data, AnnData):
        if expression is not None:
            raise ValueError("Pass `expression` only with a gene list as `data`; an AnnData brings its own counts.")
        adata, missing = data, list[str]()
    else:
        adata, missing = _panel_from_genes(data, expression)
        inplace = False

    X = _counts(adata, layer)
    gene_arm, arms = _gene_arms(_as_df(adata.var), centromeres, exclude_chromosomes)
    group = (
        pd.Categorical(["all"] * adata.n_obs)
        if groupby is None
        else pd.Categorical(np.asarray(_obs_column(adata, _obs_key(adata, groupby, "groupby"))))
    )
    if np.any(group.codes < 0):
        raise ValueError(f"`obs[{groupby!r}]` has missing values; give every cell a group.")
    is_ref = _reference(adata, reference_key, reference_cat)

    totals = np.asarray(X.sum(axis=1)).ravel()
    observed = float(np.median(totals))
    if observed <= 0:
        raise ValueError("The median cell has no counts on the panel.")
    targets = [observed] if counts_per_cell is None else list(np.asarray(counts_per_cell, dtype=float).ravel())
    if any(not t > 0 for t in targets):
        raise ValueError(f"`counts_per_cell` must be positive, got {counts_per_cell}.")

    model = _Model(gene_arm, len(arms), group.codes.astype(np.int64), len(group.categories), is_ref)
    rng = np.random.default_rng(random_state)
    frames = []
    with numba_threads(n_jobs):
        base = model.fit(X, scale=1.0)
        for target in targets:
            f = target / observed
            if np.isclose(f, 1.0):
                fit, source = base, "measured"
            elif f < 1:
                fit, source = model.fit(_thin(X, f, rng), scale=1.0), "measured"
            else:
                fit, source = model.fit(X, scale=f), "extrapolated"
                fit.null_var = 1 + (base.null_var - 1) * fit.null_spread / base.null_spread
            frames.append(_table(fit, model, group, arms, target, source, n_cells, gain, loss, alpha, power))
    table = pd.concat(frames, ignore_index=True)

    n_ref = int(is_ref.sum())
    params = _params_for_uns(
        groupby=groupby,
        reference_key=reference_key,
        reference_cat=reference_cat,
        layer=layer,
        counts_per_cell=targets,
        n_cells=n_cells,
        gain=gain,
        loss=loss,
        alpha=alpha,
        power=power,
        n_reference=n_ref,
        missing_genes=missing,
    )
    log_result(
        "detectability",
        "%d arms with panel genes, %d reference cells",
        int((table.groupby("arm")["n_genes"].first() > 0).sum()),
        n_ref,
    )
    if inplace:
        adata.uns[key_added] = {"table": table, "params": params}
        adata.obs[f"{key_added}_info"] = base.cell_info
        return None
    return table


@dataclass
class _Fit:
    """Per group and arm: mean `I`, `A`, `B` (groups, 3, arms) and the median counts; per arm the null variance."""

    moments: np.ndarray
    counts_per_cell: np.ndarray
    null_var: np.ndarray
    null_spread: np.ndarray  # sum I^2 / sum I of the reference cells, to extrapolate the null variance
    cell_info: np.ndarray


@dataclass
class _Model:
    """The genes' arms and the cells' groups, which stay the same at every depth."""

    gene_arm: np.ndarray  # arm of each gene, -1 if not used
    n_arms: int
    group: np.ndarray  # group code of each cell
    n_groups: int
    is_ref: np.ndarray

    def fit(self, X: sp.csr_matrix, *, scale: float) -> _Fit:
        """Shares, NB dispersions, and the information and null variance of every arm, on the counts `X`."""
        # each group's reference and other cells get their own expected shares, so that copy number changes of the
        # other cells don't bias the null
        code = self.group * 2 + self.is_ref
        n_codes = 2 * self.n_groups
        totals = np.asarray(X.sum(axis=1)).ravel()
        onehot = sp.csr_matrix((np.ones(X.shape[0]), (code, np.arange(X.shape[0]))), shape=(n_codes, X.shape[0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.nan_to_num(np.asarray((onehot @ X).todense()) / (onehot @ totals)[:, None])
        used = self.gene_arm >= 0
        P = np.zeros((n_codes, self.n_arms))
        for a in range(self.n_arms):
            P[:, a] = p[:, self.gene_arm == a].sum(axis=1)

        phi = self._dispersion(X, code, p, P)
        order = np.flatnonzero(used)[np.argsort(self.gene_arm[used], kind="stable")]
        bounds = np.concatenate([[0], np.cumsum(np.bincount(self.gene_arm[used], minlength=self.n_arms))])
        n_chunks = n_kernel_chunks(X.shape[0])
        sums = np.zeros((n_chunks, self.n_groups, 3, self.n_arms))
        null = np.zeros((n_chunks, 3, self.n_arms))
        cell_info = np.zeros(X.shape[0])
        _arm_information(
            X.data, X.indices, X.indptr, code, self.group, self.is_ref, P, p, phi, self.gene_arm, order, bounds,
            float(scale), sums, null, cell_info, np.empty((n_chunks, 2, self.n_arms)),
        )  # fmt: skip
        sums, null = sums.sum(axis=0), null.sum(axis=0)
        n_per_group = np.bincount(self.group, minlength=self.n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            null_var = null[0] / null[1]
            null_spread = null[2] / null[1]
        counts = np.array([np.median(totals[self.group == g]) * scale for g in range(self.n_groups)])
        return _Fit(sums / n_per_group[:, None, None], counts, null_var, null_spread, cell_info)

    def _dispersion(self, X: sp.csr_matrix, code: np.ndarray, p: np.ndarray, P: np.ndarray) -> np.ndarray:
        """NB dispersion `phi` of each gene, from `sum (x - m)^2 - m = phi sum m^2` over the reference cells."""
        rows = np.flatnonzero(self.is_ref)
        n_chunks = n_kernel_chunks(len(rows))
        n_codes, n_genes = p.shape
        xx, xm = np.zeros((n_chunks, n_genes)), np.zeros((n_chunks, n_genes))
        s1, s2 = np.zeros((n_chunks, n_codes, self.n_arms)), np.zeros((n_chunks, n_codes, self.n_arms))
        _nb_moments(
            X.data, X.indices, X.indptr, rows, code, P, p, self.gene_arm, xx, xm, s1, s2,
            np.empty((n_chunks, self.n_arms)),
        )  # fmt: skip
        arm = np.maximum(self.gene_arm, 0)
        sum_m = (p * s1.sum(axis=0)[:, arm]).sum(axis=0)
        sum_m2 = (p**2 * s2.sum(axis=0)[:, arm]).sum(axis=0)
        excess = xx.sum(axis=0) - 2 * xm.sum(axis=0) + sum_m2 - sum_m
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.clip(np.nan_to_num(excess / sum_m2), 0, _MAX_PHI)
        return np.where(self.gene_arm >= 0, phi, 0.0)


def _table(
    fit: _Fit,
    model: _Model,
    group: pd.Categorical[Any],
    arms: list[str],
    depth: float,
    source: str,
    n_cells: int,
    gain: float,
    loss: float,
    alpha: float,
    power: float,
) -> pd.DataFrame:
    """The rows of one depth: groups x arms."""
    I, A, B = (fit.moments[:, k, :] for k in range(3))
    n_genes = np.bincount(model.gene_arm[model.gene_arm >= 0], minlength=model.n_arms)
    v = np.broadcast_to(np.fmax(np.nan_to_num(fit.null_var, nan=1.0), 1.0), I.shape)
    za, zb = norm.ppf(1 - alpha), norm.ppf(power)
    out = {
        "group": np.repeat(np.asarray(group.categories), model.n_arms),
        "depth": depth,
        "counts_per_cell": np.repeat(fit.counts_per_cell, model.n_arms),
        "arm": np.tile(arms, model.n_groups),
        "chromosome": np.tile([a[:-1] for a in arms], model.n_groups),
        "n_genes": np.tile(n_genes, model.n_groups),
        "info": I.ravel(),
        "null_var": np.tile(fit.null_var, model.n_groups),
        "null_var_source": source,
    }
    for suffix, var in (("", v), ("_ideal", np.ones_like(I))):
        out[f"gain_ratio{suffix}"] = _min_ratio(I, A, B, var, n_cells, za, zb, gain=True).ravel()
        out[f"loss_ratio{suffix}"] = _min_ratio(I, A, B, var, n_cells, za, zb, gain=False).ravel()
        out[f"cells_gain{suffix}"] = _cells_needed(I, A, B, var, gain, za, zb).ravel()
        out[f"cells_loss{suffix}"] = _cells_needed(I, A, B, var, loss, za, zb).ravel()
    df = pd.DataFrame(out)
    df["group"] = pd.Categorical(df["group"], categories=group.categories)
    df["arm"] = pd.Categorical(df["arm"], categories=arms, ordered=True)
    return df


def _cells_needed(I, A, B, v, r: float, za: float, zb: float) -> np.ndarray:
    """Cells for the z-test to detect copy ratio `r`: `n = v (za sqrt(I) + zb sqrt(r A + r^2 B))^2 / ((r - 1) I)^2`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        n = v * (za * np.sqrt(I) + zb * np.sqrt(r * A + r * r * B)) ** 2 / ((r - 1) * I) ** 2
    return np.where(I > 0, n, np.inf)


def _min_ratio(I, A, B, v, n: int, za: float, zb: float, *, gain: bool) -> np.ndarray:
    """Smallest gain (largest loss ratio) that the z-test detects in `n` cells, by bisection; NaN if none.

    The margin `|r - 1| sqrt(n) I - sqrt(v) (za sqrt(I) + zb sqrt(r A + r^2 B))` is convex in `r` and negative at
    `r = 1`, so it crosses 0 at most once on each side of 1.
    """

    def margin(r):
        return np.abs(r - 1) * np.sqrt(n) * I - np.sqrt(v) * (za * np.sqrt(I) + zb * np.sqrt(r * A + r * r * B))

    lo, hi = (np.ones_like(I), np.full_like(I, _MAX_GAIN)) if gain else (np.zeros_like(I), np.ones_like(I))
    found = margin(hi if gain else lo) >= 0
    for _ in range(60):
        mid = np.sqrt(lo * hi) if gain else (lo + hi) / 2
        crossed = margin(mid) >= 0
        if gain:
            hi, lo = np.where(crossed, mid, hi), np.where(crossed, lo, mid)
        else:
            lo, hi = np.where(crossed, mid, lo), np.where(crossed, hi, mid)
    return np.where(found & (I > 0), hi if gain else lo, np.nan)


def _check_numbers(*, n_cells, gain, loss, alpha, power) -> None:
    if not (isinstance(n_cells, int | np.integer) and n_cells >= 1):
        raise ValueError(f"`n_cells` must be a positive integer, got {n_cells!r}.")
    if not gain > 1:
        raise ValueError(f"`gain` must be a copy ratio above 1, got {gain}.")
    if not 0 <= loss < 1:
        raise ValueError(f"`loss` must be a copy ratio in [0, 1), got {loss}.")
    for name, value in (("alpha", alpha), ("power", power)):
        if not 0 < value < 1:
            raise ValueError(f"`{name}` must be in (0, 1), got {value}.")


def _panel_from_genes(genes, expression: AnnData | None) -> tuple[AnnData, list[str]]:
    """The panel genes of `expression` (gene-list mode), and the genes it lacks."""
    if isinstance(genes, str) or not all(isinstance(g, str) for g in genes):
        raise TypeError("`data` must be an AnnData or a list of gene symbols.")
    if expression is None:
        raise ValueError(
            "With a gene list, pass `expression`: an AnnData of raw counts from a similar tissue, with gene positions."
        )
    genes = list(dict.fromkeys(genes))
    known = expression.var_names.isin(genes)
    missing = sorted(set(genes) - set(expression.var_names[known]))
    if not known.any():
        raise ValueError("None of the genes are in `expression.var_names`.")
    if missing:
        warnings.warn(
            f"{len(missing)} of {len(genes)} panel genes are not in `expression` and are left out: "
            f"{', '.join(missing[:10])}{', ...' if len(missing) > 10 else ''}",
            UserWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )
    return expression[:, known], missing


def _counts(adata: AnnData, layer: str | None) -> sp.csr_matrix:
    """The raw counts as a CSR matrix, after checking that they are counts."""
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"`layer={layer!r}` is not in `adata.layers`.")
    X = _as_matrix(adata.X if layer is None else adata.layers[layer], "the raw counts")
    X = sp.csr_matrix(X)
    if X.data.size and (X.data.min() < 0 or np.any(X.data != np.round(X.data))):
        raise ValueError(
            "`tl.detectability` needs raw counts (non-negative integers); pass the layer with the counts as `layer`."
        )
    return X


def _gene_arms(var: pd.DataFrame, centromeres, exclude_chromosomes) -> tuple[np.ndarray, list[str]]:
    """The arm index of each gene (-1: not used) and the arms, in genome order."""
    if not {"chromosome", "start"} <= set(var.columns):
        raise ValueError(
            "Genomic positions not found: `var` needs the columns `chromosome` and `start`, e.g. from "
            "`infercnvpy.io.genomic_position_from_gtf`."
        )
    cen = HG38_CENTROMERES if centromeres is None else dict(centromeres)
    chrom = np.asarray(var["chromosome"]).astype(str)
    start = pd.to_numeric(np.asarray(var["start"]), errors="coerce")
    known = np.isin(chrom, list(cen)) & np.isfinite(start)
    if exclude_chromosomes is not None:
        known &= ~np.isin(chrom, list(exclude_chromosomes))
    if not known.any():
        raise ValueError(
            f"No gene is on a chromosome of `centromeres` (e.g. {next(iter(cen))!r}); check the names in "
            "`var['chromosome']`."
        )
    centre = np.array([cen.get(c, np.nan) for c in chrom])
    label = np.where(known, np.char.add(chrom, np.where(start < centre, "p", "q")), "")
    with_genes = set(label[known])
    arms = [
        f"{c}{side}"
        for c in _natural_sort([c for c in cen if c not in set(exclude_chromosomes or ())])
        for side in ("p", "q")
        if f"{c}{side}" in with_genes
        or (centromeres is None and f"{c}{side}" not in _ACROCENTRIC_P)
        or (centromeres is not None and (side == "q" or cen[c] > 0))
    ]
    index = {a: i for i, a in enumerate(arms)}
    return np.array([index.get(a, -1) for a in label], dtype=np.int64), arms


def _obs_key(adata: AnnData, key: str, what: str) -> str:
    if key not in adata.obs.columns:
        raise ValueError(f"`{what}={key!r}` is not a column of `adata.obs`.")
    return key


def _reference(adata: AnnData, reference_key: str | None, reference_cat) -> np.ndarray:
    """Whether each cell is a diploid reference cell."""
    if (reference_key is None) != (reference_cat is None):
        raise ValueError("Pass both `reference_key` and `reference_cat`, or neither.")
    if reference_key is None:
        warnings.warn(
            "Measuring the null variance on all cells, which counts real copy number changes as noise. Pass "
            "`reference_key` and `reference_cat` to measure it on diploid cells.",
            UserWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )
        return np.ones(adata.n_obs, dtype=bool)
    labels = _obs_column(adata, _obs_key(adata, reference_key, "reference_key"))
    cats = _categories(reference_cat)
    if (absent := ~pd.Index(cats).isin(labels.unique())).any():
        raise ValueError(f"Reference categories not found in `obs[{reference_key!r}]`: {list(cats[absent])}")
    return np.asarray(labels.isin(cats))


def _thin(X: sp.csr_matrix, f: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Keep each count with probability `f` (binomial thinning)."""
    out = X.copy()
    out.data = rng.binomial(np.asarray(X.data, dtype=np.int64), f).astype(np.float64)
    out.eliminate_zeros()
    return out
