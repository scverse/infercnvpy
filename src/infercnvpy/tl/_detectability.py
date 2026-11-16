"""Which chromosome-arm copy number changes a gene panel and sequencing depth can detect (`tl.detectability`)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import norm

from infercnvpy._logging import SKIP_OWN_FRAMES, log_result
from infercnvpy._numba import n_kernel_chunks, numba_threads
from infercnvpy._util import _as_df, _as_matrix, _obs_column, _params_for_uns
from infercnvpy.tl._copy_number import _as_list, _reference_cells
from infercnvpy.tl._detectability_kernels import _arm_information, _nb_moments
from infercnvpy.tl._infercnv import _group_sums, _kernel_ready, _natural_sort

#: GRCh38 centromere midpoints (bp), from the "acen" bands of the UCSC cytoband table
HG38_CENTROMERES = {
    c: mb * 1e6
    for c, mb in {
        "chr1": 123.4, "chr2": 93.9, "chr3": 90.9, "chr4": 50.0, "chr5": 48.8, "chr6": 59.8, "chr7": 60.1,
        "chr8": 45.2, "chr9": 43.0, "chr10": 39.8, "chr11": 53.4, "chr12": 35.5, "chr13": 17.7, "chr14": 17.2,
        "chr15": 19.0, "chr16": 36.8, "chr17": 25.1, "chr18": 18.5, "chr19": 26.2, "chr20": 28.1, "chr21": 12.0,
        "chr22": 15.0, "chrX": 60.6, "chrY": 10.4,
    }.items()
}  # fmt: skip
#: p arms of the acrocentric chromosomes, which hold no genes of their own: listed only when the panel has genes there
_ACROCENTRIC_P = {"chr13p", "chr14p", "chr15p", "chr21p", "chr22p"}
_MAX_PHI = 100.0  # cap of the NB dispersion 1 / k of a gene
_MIN_REF_COUNTS = 20  # reference counts of a gene below which its dispersion is not trusted on its own


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

    For every chromosome arm, the function reports the smallest gain and loss that a test on `n_cells` cells
    detects, and the number of cells it needs to detect a `gain` (default 1.5x) and a `loss` (default 0.5x), from the
    expression and depth of the data. Use it before an experiment, with a gene list and expression from a similar
    dataset, to choose a panel and depth. Use it after the experiment, on the data itself, to see on which arms the
    CNV calls can be trusted.

    Each number comes twice, as an ideal bound from counting noise alone and as the headline "achievable" value. The
    achievable value also includes the extra variance that diploid cells show on the arm. On small marker panels, cell
    type and cell state programs move the few genes of an arm together, which looks like copy number. The ideal bound
    ignores these programs and over-promises, and the gap between the two columns shows by how much.

    **Model.** A diploid cell expects `m = s p` counts of a gene, with `p` the gene's share of the counts of the cell's
    group (`groupby`) and `s` the cell's size. The size is the cell's count off the arm, divided by the share of the
    genes off the arm, so that the arm's own counts don't enter. Counts are negative binomial with mean `m` and
    dispersion `phi = 1 / k` (size `k`), fitted per gene by the method of moments on the reference cells. A copy ratio
    `r` of the arm multiplies `m` by `r`. Without a change, the cell's efficient score for the arm,
    `U = sum (x - m) / (1 + m phi)` over the arm's genes, has mean 0 and variance `I = sum m / (1 + m phi)`. `I` is the
    Fisher information about the log copy ratio of the negative binomial with a log link :cite:`Lawless1987`, the
    working weight that DESeq2 uses for its Wald test :cite:`Love2014`. With a change, `U` has mean `(r - 1) I` and
    variance `r A + r^2 B`, where `A` and `B` are the parts of `I` from counting and from dispersion.

    **Null variance.** Genes of an arm co-vary within cells (cell type and state programs), which the model misses.
    `null_var` measures the variance of the score on the reference cells as `sum U^2 / sum I` per arm, with each
    reference cell left out of the shares that give its expectation. Counting noise alone gives 1, and the noise of
    each cell's size factor adds a little (0.1 to 0.3 in simulations). The achievable values divide the information
    by the null variance, and the ideal values use 1. Give the cell types as `groupby`, so that differences between
    types are part of the expectation and not of the null variance. A group without reference cells gets its expected
    expression from its own cells and the null variance of the other groups' reference cells.

    Genes with fewer than 20 reference counts get at least the median dispersion of the better measured genes, so that
    a gene the reference hardly expresses doesn't count as a Poisson gene. An arm with panel genes but no information
    in the reference cells has no measured null variance, and its achievable values are NaN, with a warning.

    **Test.** The test is a one-sided z-test of the score summed over `n` cells, at level `alpha` and with `power`
    :cite:`Cox1974`. With `I`, `A` and `B` averaged over the cells of a group and `v` the null variance, it detects
    ratio `r` when `|r - 1| sqrt(n) I >= sqrt(v) (z_alpha sqrt(I) + z_power sqrt(r A + r^2 B))`. Solved for `n`, the
    condition gives the cells needed, and solved for `r` the smallest detectable ratio. The test treats cells as
    independent, so a program shared by all cells of a clone (a tumour state, for example) is not averaged away by
    more cells. It is the bound for a method that knows the diploid expression of every cell type, and methods that
    estimate the expression from the data need more cells.

    **Calibration.** We injected arm gains of 1.5x and losses of 0.5x into normal cells of an Atera breast sample
    (fibroblasts, normal epithelium and one tumour clone), cut to five panels at two depths. An ideal score test with
    the true expected expression needed 0.6 to 1.1 times the cells of the achievable column, and the two rank the arms
    alike (Spearman 0.98 to 0.99). The ideal column over-promises by up to about 3 times on deep, wide panels, where
    the null variance is largest. :func:`infercnvpy.tl.infercnv` needed more cells than the achievable column, by a
    factor that depends on the panel:

    ===============================  ==============================
    panel                            cells relative to `cells_gain`
    ===============================  ==============================
    whole transcriptome              about 2 to 5 times
    Xenium 5K                        about 2 to 6 times
    1,000 genes or fewer             about 3 to 8 times
    ===============================  ==============================

    On panels of 1,000 genes or fewer, `tl.infercnv` missed about a third of the arm events at any number of cells.
    When many arms change at once, it also shifts arms without a change, by smoothing over arm borders and by
    centring each cell. :func:`infercnvpy.tl.copy_number` detects clones rather than single arms, so the table
    doesn't apply to it, and on panels below its guard it labels the cells "undetermined".

    **Depths.** With `counts_per_cell`, the counts are thinned at random to each target median per cell, and the null
    variance and information are measured again with the same expected shares and dispersions. A target above the
    observed depth scales the expected counts up instead (`null_var_source="extrapolated"`). The null variance then
    has two parts. The noise of the size factor follows from the model at the new depth, and the excess of the cell
    programs grows in proportion to the information.

    **Gene-list mode.** Pass a list of gene symbols as `data` and an AnnData of raw counts from a similar tissue as
    `expression`. The panel is then cut out of `expression`, and the result describes the cells of `expression` at
    its depth on the panel. Set the depth of your platform with `counts_per_cell`, since imaging panels often have more
    counts per gene than a droplet dataset.

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
        measured on these cells. Default: all cells, which counts real copy number changes as null variance and makes
        the result conservative.
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
        Power of the test, at least 0.5.
    centromeres
        Centromere position (bp) of each chromosome, which splits it into the p and q arm. Default: GRCh38.
    exclude_chromosomes
        Chromosomes to leave out.
    random_state
        Seed of the thinning for `counts_per_cell`. Each target depth gets its own stream, so the rows of one depth
        don't depend on the other depths in the list.
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
        One row per depth (in the order of `counts_per_cell`), group (in the order of its categories) and arm (in
        genome order), with the columns

        * `group`, the cell type (`"all"` without `groupby`);
        * `depth`, the median counts per cell of all cells (the target of `counts_per_cell`);
        * `counts_per_cell`, the median counts per cell of the group;
        * `arm` and `chromosome`, the arm (`"chr1p"`, for example) and its chromosome;
        * `n_genes`, the panel genes on the arm. Arms with fewer than about 5 genes give fragile results;
        * `info`, the Fisher information per cell about the log copy ratio of the arm (mean over the group's cells);
        * `null_var`, the variance of the arm's score in reference cells relative to the model, at least 1 (1 means
          counting noise only, NaN means not measured);
        * `null_var_source`, `"measured"` (on the counts, thinned to the depth) or `"extrapolated"` (a depth above
          the data's);
        * `gain_ratio` and `loss_ratio`, the smallest gain and mildest loss (copy ratios) detectable in `n_cells`
          cells, NaN if no gain or no loss down to 0 is detectable;
        * `cells_gain` and `cells_loss`, the cells needed to detect a `gain` and a `loss` (`inf` without information);
        * the same four with the suffix `_ideal`, the counting bound without the null variance.
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

    Before an experiment, for a panel at 100 and at 200 counts per cell, with expression from a droplet atlas of the
    tissue:

    >>> cnv.tl.detectability(panel_genes, expression=atlas, groupby="cell_type", counts_per_cell=[100, 200])
    """
    _check_numbers(n_cells=n_cells, gain=gain, loss=loss, alpha=alpha, power=power)
    if isinstance(data, AnnData):
        if expression is not None:
            raise ValueError("Pass `expression` only with a gene list as `data`; an AnnData brings its own counts.")
        adata = data
    else:
        adata = _panel_from_genes(data, expression)
        inplace = False
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"`tl.detectability` needs cells and genes, got {adata.n_obs} x {adata.n_vars}.")

    X = _counts(adata, layer)
    exclude = [] if exclude_chromosomes is None else _as_list(exclude_chromosomes)
    gene_arm, arms = _gene_arms(_as_df(adata.var), centromeres, exclude)
    group = _groups(adata, groupby)
    is_ref = _reference(adata, reference_key, reference_cat)

    totals = np.asarray(X.sum(axis=1)).ravel()
    observed = float(np.median(totals))
    if observed <= 0:
        raise ValueError("The median cell has no counts on the panel.")
    targets = [observed] if counts_per_cell is None else np.asarray(counts_per_cell, dtype=float).ravel().tolist()
    if not targets or not all(np.isfinite(t) and t > 0 for t in targets):
        raise ValueError(f"`counts_per_cell` must be one or more positive numbers, got {counts_per_cell!r}.")

    model = _Model(gene_arm, len(arms), group.codes.astype(np.int64), len(group.categories), is_ref)
    frames = []
    with numba_threads(n_jobs):
        model.fit(X)
        base = model.run(X, 1.0) if inplace or any(t >= observed or np.isclose(t, observed) for t in targets) else None
        for target in targets:
            f = target / observed
            if np.isclose(f, 1.0):
                assert base is not None
                fit, null_var, source = base, _measured(base), "measured"
            elif f < 1:
                rng = np.random.default_rng([random_state, int(round(target * 1000))])
                fit = model.run(_thin(X, f, rng), 1.0)
                null_var, source = _measured(fit), "measured"
            else:
                assert base is not None
                fit = model.run(X, f)
                null_var, source = _extrapolated(base, fit), "extrapolated"
            frames.append(_table(fit, null_var, model, group, arms, target, source, n_cells, gain, loss, alpha, power))
    table = pd.concat(frames, ignore_index=True)
    _warn_unmeasured(table)

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
        centromeres=None if centromeres is None else dict(centromeres),
        exclude_chromosomes=exclude,
        random_state=random_state,
        n_reference=n_ref,
    )
    log_result("detectability", "%d arms with panel genes, %d reference cells", int((model.n_genes > 0).sum()), n_ref)
    if inplace:
        assert base is not None
        adata.uns[key_added] = {"table": table, "params": params}
        adata.obs[f"{key_added}_info"] = base.cell_info
        return None
    return table


@dataclass
class _Fit:
    """Per group and arm the mean `I`, `A`, `B` (groups, 3, arms) and the median counts, and the null sums of `run`."""

    moments: np.ndarray
    counts_per_cell: np.ndarray
    null: np.ndarray  # (4, arms): sums of U^2, I, I^2 and the size-factor part of U^2 over the reference cells
    cell_info: np.ndarray


@dataclass
class _Model:
    """The genes' arms and the cells' groups, and the expected shares and dispersions fitted once by `fit`."""

    gene_arm: np.ndarray  # arm of each gene, -1 if not used
    n_arms: int
    group: np.ndarray  # group code of each cell
    n_groups: int
    is_ref: np.ndarray
    # set by `fit`
    p: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    P2: np.ndarray = field(init=False)
    Ns: np.ndarray = field(init=False)
    phi: np.ndarray = field(init=False)

    def __post_init__(self):
        # each group's reference and other cells get their own expected shares, so that copy number changes of the
        # other cells don't bias the null
        self.code = self.group * 2 + self.is_ref
        used = self.gene_arm >= 0
        self.order = np.flatnonzero(used)[np.argsort(self.gene_arm[used], kind="stable")]
        self.pos = np.full(self.gene_arm.size, -1, dtype=np.int64)
        self.pos[self.order] = np.arange(self.order.size)
        self.n_genes = np.bincount(self.gene_arm[used], minlength=self.n_arms)
        self.bounds = np.concatenate([[0], np.cumsum(self.n_genes)])
        self.to_arm = sp.csr_matrix(
            (np.ones(used.sum()), (np.flatnonzero(used), self.gene_arm[used])), shape=(self.gene_arm.size, self.n_arms)
        )

    def fit(self, X: sp.csr_matrix) -> None:
        """Expected shares of each share group and the NB dispersion of each gene, on the observed counts `X`."""
        n_codes = 2 * self.n_groups
        sums = _group_sums(X, np.arange(X.shape[0]), self.code, n_codes)
        self.Ns = sums.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.p = np.nan_to_num(sums / self.Ns[:, None])
        self.P = np.asarray(self.p @ self.to_arm)
        self.phi = self._dispersion(X, ref_counts=sums[1::2].sum(axis=0))
        q = self.p**2 * self.phi
        self.P2 = np.column_stack([np.asarray(q @ self.to_arm), q.sum(axis=1)])

    def run(self, X: sp.csr_matrix, scale: float) -> _Fit:
        """Information of every group and arm, and the null sums of the reference cells, on the counts `X`."""
        n_chunks = n_kernel_chunks(X.shape[0])
        sums = np.zeros((n_chunks, self.n_groups, 3, self.n_arms))
        null = np.zeros((n_chunks, 4, self.n_arms))
        cell_info = np.zeros(X.shape[0])
        _arm_information(
            X.data, X.indices, X.indptr, self.code, self.group, self.is_ref, self.P, self.P2,
            np.ascontiguousarray(self.p[:, self.order]), np.ascontiguousarray(self.phi[self.order]), self.pos,
            self.Ns, self.gene_arm, self.bounds, float(scale), sums, null, cell_info,
            np.empty((n_chunks, 2, self.n_arms)),
        )  # fmt: skip
        totals = np.asarray(X.sum(axis=1)).ravel() * scale
        counts = np.array([np.median(totals[self.group == g]) for g in range(self.n_groups)])
        n_per_group = np.bincount(self.group, minlength=self.n_groups)
        return _Fit(sums.sum(axis=0) / n_per_group[:, None, None], counts, null.sum(axis=0), cell_info)

    def _dispersion(self, X: sp.csr_matrix, ref_counts: np.ndarray) -> np.ndarray:
        """NB dispersion of each gene, `sum (x - m)^2 - m = phi sum m^2` over the reference cells (leave-one-out m).

        Genes with fewer than `_MIN_REF_COUNTS` reference counts get at least the median dispersion of the others.
        """
        rows = np.flatnonzero(self.is_ref)
        n_chunks = n_kernel_chunks(len(rows))
        n_codes, n_genes = self.p.shape
        xx, xm, dm, dm2 = (np.zeros((n_chunks, n_genes)) for _ in range(4))
        s1, s2 = np.zeros((n_chunks, n_codes, self.n_arms)), np.zeros((n_chunks, n_codes, self.n_arms))
        _nb_moments(
            X.data, X.indices, X.indptr, rows, self.code, self.P, self.p, self.Ns, self.gene_arm, xx, xm, dm, dm2,
            s1, s2, np.empty((n_chunks, self.n_arms)),
        )  # fmt: skip
        arm = np.maximum(self.gene_arm, 0)
        sum_m = (self.p * s1.sum(axis=0)[:, arm]).sum(axis=0) + dm.sum(axis=0)
        sum_m2 = (self.p**2 * s2.sum(axis=0)[:, arm]).sum(axis=0) + dm2.sum(axis=0)
        excess = xx.sum(axis=0) - 2 * xm.sum(axis=0) + sum_m2 - sum_m
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.clip(np.nan_to_num(excess / sum_m2), 0, _MAX_PHI)
        used = self.gene_arm >= 0
        well = used & (ref_counts >= _MIN_REF_COUNTS)
        pooled = float(np.median(phi[well])) if well.any() else 1.0
        phi = np.where(ref_counts >= _MIN_REF_COUNTS, phi, np.maximum(phi, pooled))
        return np.where(used, phi, 0.0)


def _measured(fit: _Fit) -> np.ndarray:
    """Null variance per arm, `sum U^2 / sum I` over the reference cells, at least 1 (NaN without information)."""
    U2, I, _, _ = fit.null
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.fmax(U2 / I, 1.0) + np.where(I > 0, 0.0, np.nan)


def _extrapolated(base: _Fit, fit: _Fit) -> np.ndarray:
    """Null variance at a depth above the observed one (`fit`), from its measurement at the observed depth (`base`).

    The size-factor part comes from the model at the new depth. The rest, from cell programs, grows like
    `sum I^2 / sum I` (the score variance of a program on the arm grows with the square of the information).
    """
    U2, I, I2, C = base.null
    _, If, I2f, Cf = fit.null
    with np.errstate(invalid="ignore", divide="ignore"):
        programs = np.fmax(U2 / I - 1 - C / I, 0.0)
        return 1 + Cf / If + programs * (I2f / If) / (I2 / I)


def _table(
    fit: _Fit,
    null_var: np.ndarray,
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
    I, A, B = fit.moments[:, 0], fit.moments[:, 1], fit.moments[:, 2]
    za, zb = norm.ppf(1 - alpha), norm.ppf(power)
    out = {
        "group": pd.Categorical(np.repeat(group.categories, model.n_arms), categories=group.categories),
        "depth": depth,
        "counts_per_cell": np.repeat(fit.counts_per_cell, model.n_arms),
        "arm": pd.Categorical(np.tile(arms, model.n_groups), categories=arms, ordered=True),
        "chromosome": np.tile([a[:-1] for a in arms], model.n_groups),
        "n_genes": np.tile(model.n_genes, model.n_groups),
        "info": I.ravel(),
        "null_var": np.tile(null_var, model.n_groups),
        "null_var_source": source,
    }
    for suffix, v in (("", null_var), ("_ideal", 1.0)):
        out[f"gain_ratio{suffix}"] = _min_ratio(I, A, B, v, n_cells, za, zb, gain=True).ravel()
        out[f"loss_ratio{suffix}"] = _min_ratio(I, A, B, v, n_cells, za, zb, gain=False).ravel()
        out[f"cells_gain{suffix}"] = _cells_needed(I, A, B, v, gain, za, zb).ravel()
        out[f"cells_loss{suffix}"] = _cells_needed(I, A, B, v, loss, za, zb).ravel()
    return pd.DataFrame(out)


def _cells_needed(I, A, B, v, r: float, za: float, zb: float) -> np.ndarray:
    """Cells for the z-test to detect copy ratio `r`: `n = v (za sqrt(I) + zb sqrt(r A + r^2 B))^2 / ((r - 1) I)^2`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        n = v * (za * np.sqrt(I) + zb * np.sqrt(r * A + r * r * B)) ** 2 / ((r - 1) * I) ** 2
    return np.where(I > 0, n, np.inf)


def _min_ratio(I, A, B, v, n: int, za: float, zb: float, *, gain: bool) -> np.ndarray:
    """Smallest gain (largest loss ratio) that the z-test detects in `n` cells; NaN if none.

    With `r = 1 + s u` (`s = 1` for a gain, `-1` for a loss) and `k = sqrt(n) I`, `c = sqrt(v) za sqrt(I)`, the test
    detects `r` when `k u - c >= sqrt(v) zb sqrt(r A + r^2 B)`. With `zb >= 0` (power at least 0.5) the left side has
    to be positive, and squaring gives the quadratic `(k^2 - v zb^2 B) u^2 - (2 k c + s v zb^2 (A + 2 B)) u
    + c^2 - v zb^2 I = 0`, whose root above `c / k` is the answer.
    """
    s = 1.0 if gain else -1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        k, c, q = np.sqrt(n) * I, np.sqrt(v) * za * np.sqrt(I), v * zb**2
        a2, a1, a0 = k**2 - q * B, -(2 * k * c + s * q * (A + 2 * B)), c**2 - q * I
        disc = np.sqrt(a1**2 - 4 * a2 * a0)
        roots = np.stack([(-a1 - disc) / (2 * a2), (-a1 + disc) / (2 * a2)])
        roots = np.where(np.isclose(a2, 0), -a0 / a1, roots)
        roots = np.where((roots >= c / k * (1 - 1e-9)) & (gain | (roots <= 1)) & (I > 0), roots, np.nan)
    return 1 + s * np.fmin(roots[0], roots[1])


def _check_numbers(*, n_cells, gain, loss, alpha, power) -> None:
    if not (isinstance(n_cells, int | np.integer) and n_cells >= 1):
        raise ValueError(f"`n_cells` must be a positive integer, got {n_cells!r}.")
    if not gain > 1:
        raise ValueError(f"`gain` must be a copy ratio above 1, got {gain}.")
    if not 0 <= loss < 1:
        raise ValueError(f"`loss` must be a copy ratio in [0, 1), got {loss}.")
    if not 0 < alpha < 1:
        raise ValueError(f"`alpha` must be in (0, 1), got {alpha}.")
    if not 0.5 <= power < 1:
        raise ValueError(f"`power` must be in [0.5, 1), got {power}.")


def _panel_from_genes(genes, expression: AnnData | None) -> AnnData:
    """The panel genes of `expression` (gene-list mode)."""
    if isinstance(genes, str):
        raise TypeError("`data` must be an AnnData or a list of gene symbols.")
    genes = list(dict.fromkeys(genes))
    if not all(isinstance(g, str) for g in genes):
        raise TypeError("`data` must be an AnnData or a list of gene symbols.")
    if expression is None:
        raise ValueError(
            "With a gene list, pass `expression`, an AnnData of raw counts from a similar tissue with gene positions."
        )
    known = expression.var_names.isin(genes)
    if not known.any():
        raise ValueError("None of the genes are in `expression.var_names`.")
    if not expression.var_names[known].is_unique:
        raise ValueError("Some panel genes occur more than once in `expression.var_names`; make the names unique.")
    if missing := sorted(set(genes) - set(expression.var_names[known])):
        warnings.warn(
            f"{len(missing)} of {len(genes)} panel genes are not in `expression` and are left out: "
            f"{', '.join(missing[:10])}{', ...' if len(missing) > 10 else ''}",
            UserWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )
    return expression[:, known]


def _counts(adata: AnnData, layer: str | None) -> sp.csr_matrix:
    """The raw counts as a canonical CSR matrix, after checking that they are counts."""
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"`layer={layer!r}` is not in `adata.layers`.")
    X = _kernel_ready(sp.csr_matrix(_as_matrix(adata.X if layer is None else adata.layers[layer], "the raw counts")))
    if X.data.size and (X.data.min() < 0 or np.any(X.data != np.round(X.data))):
        raise ValueError(
            "`tl.detectability` needs raw counts (non-negative integers); pass the layer with the counts as `layer`."
        )
    return X


def _gene_arms(var: pd.DataFrame, centromeres, exclude: list[str]) -> tuple[np.ndarray, list[str]]:
    """The arm index of each gene (-1: not used) and the arms, in genome order.

    Arms without panel genes are listed too, except the p arms of acrocentric chromosomes and of chromosomes whose
    centromere is at position 0.
    """
    if not {"chromosome", "start"} <= set(var.columns):
        raise ValueError(
            "Genomic positions not found: `var` needs the columns `chromosome` and `start`, e.g. from "
            "`infercnvpy.io.genomic_position_from_gtf`."
        )
    cen = {c: pos for c, pos in (HG38_CENTROMERES if centromeres is None else centromeres).items() if c not in exclude}
    chrom = np.asarray(var["chromosome"]).astype(str)
    start = pd.to_numeric(np.asarray(var["start"]), errors="coerce")
    centre = np.array([cen.get(c, np.nan) for c in chrom])
    known = np.isfinite(centre) & np.isfinite(start)
    if not known.any():
        raise ValueError("No gene is on a chromosome of `centromeres`; check the names in `var['chromosome']`.")
    label = np.where(known, np.char.add(chrom, np.where(start < centre, "p", "q")), "")
    skip = _ACROCENTRIC_P | {f"{c}p" for c, pos in cen.items() if pos <= 0}
    present = set(label[known])
    arms = [f"{c}{s}" for c in _natural_sort(list(cen)) for s in "pq" if f"{c}{s}" in present or f"{c}{s}" not in skip]
    return pd.Index(arms).get_indexer(pd.Index(label)).astype(np.int64), arms


def _groups(adata: AnnData, groupby: str | None) -> pd.Categorical[Any]:
    """The group of each cell, with the categories in the order of `obs[groupby]`."""
    if groupby is None:
        return pd.Categorical(np.full(adata.n_obs, "all"))
    if groupby not in adata.obs.columns:
        raise ValueError(f"`groupby={groupby!r}` is not a column of `adata.obs`.")
    col = _obs_column(adata, groupby).astype("category").cat.remove_unused_categories()
    if col.isna().any():
        raise ValueError(f"`obs[{groupby!r}]` has missing values; give every cell a group.")
    return pd.Categorical(col)


def _reference(adata: AnnData, reference_key: str | None, reference_cat) -> np.ndarray:
    """Whether each cell is a diploid reference cell."""
    is_ref = _reference_cells(adata, reference_key, reference_cat)
    if is_ref is None:
        warnings.warn(
            "Measuring the null variance on all cells, which counts real copy number changes as noise. Pass "
            "`reference_key` and `reference_cat` to measure it on diploid cells.",
            UserWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )
        return np.ones(adata.n_obs, dtype=bool)
    if not is_ref.any():
        raise ValueError(f"`reference_cat={reference_cat!r}` selects no cells.")
    return is_ref


def _warn_unmeasured(table: pd.DataFrame) -> None:
    unmeasured = table.loc[(table["n_genes"] > 0) & table["null_var"].isna(), "arm"].astype(str).unique()
    if len(unmeasured):
        warnings.warn(
            f"The reference cells have no information on the arms {', '.join(unmeasured)}, so their null variance "
            "is not measured and their achievable values are NaN.",
            UserWarning,
            skip_file_prefixes=SKIP_OWN_FRAMES,
        )


def _thin(X: sp.csr_matrix, f: float, rng: np.random.Generator, chunk: int = 1 << 20) -> sp.csr_matrix:
    """Keep each count with probability `f` (binomial thinning), sharing the structure of `X` (zeros stay stored)."""
    data = np.empty_like(X.data)
    for s in range(0, data.size, chunk):
        data[s : s + chunk] = rng.binomial(X.data[s : s + chunk].astype(np.int64), f)
    out = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape, copy=False)
    out.has_canonical_format = True
    return out
