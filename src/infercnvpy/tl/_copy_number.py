"""Copy number profiles, clones and aneuploid/diploid labels from raw counts with a clone-level count model.

The method is infercnvpy's own; see the docstring of `copy_number` for the steps. Implementation notes:

- the HMM is a forward-backward pass per profile in numba (`_copy_number_kernels.hmm_posteriors`), with Student-t
  emissions on gene-level log ratios and transitions that depend on the genomic distance between genes;
- per-cell work (window counts, pooled counts, likelihoods, dispersion moments) are fused numba kernels over row blocks
  of the count matrix, so memory is linear in the number of cells: one float32 matrix of cells x windows of genes;
- constants were chosen on the GSE148673 samples and the cnv-bench datasets (Schmid et al. 2025); the method targets
  droplet (10x) and imaging data with UMI counts, not full-length plate protocols.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from infercnvpy._logging import log, log_result
from infercnvpy._numba import n_kernel_chunks, numba_threads
from infercnvpy._util import _as_matrix, _params_for_uns
from infercnvpy.tl import _copy_number_kernels as K
from infercnvpy.tl._infercnv import _categories, _natural_sort, _Rows

#: Cell-cycle genes of Tirosh et al. 2016 (S and G2/M phase), with the current symbols of renamed genes added.
CELL_CYCLE_GENES = (
    # S phase
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
    "UHRF1", "MLF1IP", "CENPU", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
    "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
    # G2/M phase
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67",
    "TMPO", "CENPF", "TACC3", "FAM64A", "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11",
    "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C",
    "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR",
    "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
)  # fmt: skip

#: GRCh38 chromosome lengths, so that bins are the same for every dataset
HG38_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
    "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
    "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}  # fmt: skip

_KMEANS_ITER = 20

LABELS = ("aneuploid", "diploid", "not.defined")
UNDETERMINED = "undetermined"

#: Markers of immune, endothelial and stromal cells, which are rarely tumour cells
NORMAL_MARKERS = (
    "PTPRC", "CD3E", "CD3D", "CD2", "CD7", "NKG7", "GNLY", "MS4A1", "CD79A", "CD19", "CD14", "LYZ", "CD68", "CSF1R",
    "FCGR3A", "TYROBP", "FCER1G", "C1QA", "C1QB", "CPA3", "TPSAB1", "PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "COL1A1",
    "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "RGS5",
)  # fmt: skip

#: copy ratios of the HMM states (expression log ratios are their logarithms)
STATE_RATIOS = (0.35, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6)
_P_NEUTRAL = 0.5  # prior probability of the neutral state
_NU = 1.5  # degrees of freedom of the t emissions
_N_HVG = 2000
_N_PCS = 30
_N_SUB = 20000  # cells for the expression PCA
_BLOCK = 25  # genes per block of the genomic statistics (flatness, smooth amplitude, mode); fewer on small panels
_MIN_CLONE = 5
_EXACT_SVD = 10_000  # cells up to which the expression PCA uses an SVD (above: eigendecomposition)
_DOSAGE_PRIOR = 1.0  # strength of the shrinkage of dosage weights towards 1, in median gene precisions
_MIN_PRIOR = 0.01  # smallest prior clone probability of a cell type
_AMP_TUMOUR = 0.1  # smooth amplitude (log ratio) of an aneuploid clone's profile, see `_smooth_amplitude`
_N_PASSES = 4
_RESEED_GAP = 0.05  # how much flatter than the diploid clone another clone must be to restart the passes from it
_FLAT_GAP = 0.15  # difference of the cluster means of group flatness that counts as a clear split
_MARKER_FOLD = 5.0
_MARKER_MIN = 1e-3  # share of counts in the markers above which a group can be marker-rich
_MAX_SPILL = 0.5  # largest spillover fraction of tumour transcripts in a normal cell's diploid expectation
_MIN_SEED = 3  # cells of the smallest group that can be the first diploid group (a rare normal population)
_MIN_RESEED = 10  # cells of the smallest clone that can become the diploid clone later
_MIN_SEG_GENES = 20  # genes of the shortest run that counts towards the burden; fewer on small panels
_MIN_EFFECT = 0.1  # smallest posterior mean log ratio that counts as a copy number change
_MIN_PANEL_GENES = 1000  # genes passing the filters below which the calls are undetermined (small panels)
_MIN_MEDIAN_COUNTS = 150  # median counts per cell in those genes below which the calls are undetermined


def copy_number(
    adata: AnnData,
    *,
    layer: str | None = None,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    exclude_chromosomes: Sequence[str] | None = ("chrY",),
    exclude_genes: Sequence[str] | None = None,
    min_genes: int = 200,
    min_detection: float = 0.05,
    window_size: int = 10,
    n_clusters: int | None = None,
    segment_length: float = 30e6,
    min_burden: float = 0.03,
    normal_markers: Sequence[str] | Mapping[str, Sequence[str]] | None = NORMAL_MARKERS,
    n_iter: int = 10,
    calculate_gene_values: bool = False,
    max_fit_cells: int | None = None,
    connectivity_key: str | None = None,
    n_hops: int = 1,
    force_calls: bool = False,
    bin_size: int = 1_000_000,
    chunksize: int = 5000,
    random_state: int = 0,
    key_added: str = "copy_number",
    inplace: bool = True,
    n_jobs: int | None = None,
) -> None | tuple[pd.DataFrame, np.ndarray, dict, np.ndarray | None]:
    """Infer copy number profiles, clones and aneuploid/diploid labels with a clone-level count model.

    The expected count of a gene in a cell is the cell's size factor times a per-gene baseline (pooled diploid
    cells) times the expression deviation of the cell's type times the copy ratio of the cell's clone, with negative
    binomial noise. The steps are:

    1. keep cells with at least `min_genes` detected genes and genes detected in at least `min_detection` of them
       (on chromosomes named like "chr1"); sum each cell's counts in windows of `window_size` genes;
    2. cluster cells by expression (k-means on principal components of the most variable genes); these clusters
       are the expression types. A copy number HMM on the difference of the pooled profiles of every pair of types
       groups the types that differ in expression only;
    3. take the first baseline from the groups whose pooled profile is flat along the genome, and from groups that
       express `normal_markers` far above the rest (or from `reference_cat`);
    4. the expression deviation of a type is the gene-level part of its pooled profile that the HMM does not
       explain, so it holds the cell type but no copy number signal. A clone's profile is an HMM (8 copy-ratio states,
       transitions by genomic distance, `segment_length`) on the gene-level log ratio of its pooled counts to their
       expectation, corrected for the cell's capture of highly expressed genes and centred on its most common level;
    5. assign cells to clones by EM on the negative binomial likelihood of their window counts. Profiles are
       cross-fitted (windows of one parity are scored with the profile fitted on the other parity), clones without
       differences merge, and clones whose likelihood gain does not pay for their segments are dropped;
    6. repeat: a clearly flatter clone becomes diploid (unless normal markers chose the baseline), aneuploid clones
       without a smooth profile (amplitude below 0.1) merge into the diploid clone, which re-defines the baseline;
    7. learn per-gene dosage weights from the clone contrasts and per-type clone priors from a non-negative fit of
       each type's profile on the clone profiles; the per-cell posterior uses these priors and log likelihoods scaled
       by a temperature calibrated between the two halves of the windows.

    The method is made for UMI counts from droplet (10x) and imaging platforms. On full-length plate data
    (Smart-seq2) its profiles are poor.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts and genomic positions (`chromosome`, `start` and `end` in `adata.var`).
    layer
        Layer with the raw counts. Default: `adata.X`.
    reference_key
        Column of `adata.obs` with annotations of normal cells. With `reference_cat`, these cells define the baseline.
        Providing reference cells improves the detection of small clones on dense panels.
    reference_cat
        One or more values of `adata.obs[reference_key]` that annotate normal cells.
    exclude_chromosomes
        Chromosomes to leave out.
    exclude_genes
        Genes to leave out. Default: HLA and cell-cycle genes (:data:`~infercnvpy.tl._copy_number.CELL_CYCLE_GENES`,
        after :cite:`Tirosh2016melanoma`).
    min_genes
        Minimum number of detected genes of a cell.
    min_detection
        Minimum fraction of cells in which a gene is detected.
    window_size
        Genes per window of the per-cell likelihood.
    n_clusters
        Number of expression clusters. Default: one per 50 cells, between 8 and 40.
    segment_length
        Expected distance between copy number changes, in bases (sets the HMM transition probabilities).
    min_burden
        Fraction of genes in non-neutral states below which two profiles count as the same copy number.
    normal_markers
        Marker genes of cell types that are rarely tumour cells (a list, or a dict of lists per cell type), matched
        against `adata.var_names`. Expression groups whose share of counts in these genes is far above that of the
        marker-poor groups (and that are flatter than the median group) join the first diploid baseline, next to the
        groups that are flattest along the genome, and keep the baseline from moving to a flatter clone later.
        Marker-rich expression types also keep their own smooth deviation from the baseline in their diploid
        expectation, so that a normal type whose expression differs smoothly along the genome is not taken for
        aneuploid. This is prior knowledge, like the marker step of SCEVAN :cite:`DeFalco2021`; it matters in samples
        with few normal cells. Default: immune, endothelial and stromal markers
        (:data:`~infercnvpy.tl._copy_number.NORMAL_MARKERS`). `None` uses the genomic flatness alone.
    n_iter
        Maximum number of EM iterations.
    max_fit_cells
        Fit the model (expression types, clones, baseline) on a random sample of this many cells and then assign all
        cells with it in one pass over the counts. Faster and leaner for very large datasets; results can differ from
        the default, which fits on all cells. Use tens of thousands of cells: with a few thousand or fewer, the
        diploid baseline and the clones can change from those of the full fit.
    connectivity_key
        Smooth the results over a spatial neighbour graph, the cells x cells graph `adata.obsp[connectivity_key]`
        (e.g. `"spatial_connectivities"` from :func:`squidpy.gr.spatial_neighbors_knn`), as
        :func:`~infercnvpy.tl.smooth_cnv` does for a CNV matrix. Each cell's posterior clone probabilities and llr
        become the mean over the cell and its neighbours (weighted by the graph); labels, clones and profiles follow
        from the smoothed posteriors with the same rules. On imaging data, where single cells hold few counts and
        neighbours usually share their clone, this lowers the errors inside clones and the false calls in normal
        tissue, at the cost of blurring the boundary between clones by about `n_hops` neighbours. About 10 neighbours
        (`squidpy.gr.spatial_neighbors_knn(adata, n_neighs=10)`) kept boundaries best on simulated clones; 100
        blurred them more than no smoothing. Across cell types, smoothing also spreads tumour calls into the normal
        cells next to a tumour. To avoid that, set the graph's entries between cells of different types to 0; this
        suits tumours that follow a cell type, while the full graph suits clones that span types.
        Default `None`: no smoothing.
    n_hops
        Number of smoothing steps over the graph of `connectivity_key`; each step reaches one more edge away.
    force_calls
        Call cells even where the data hold too little copy number information. With fewer than 1,000 genes passing
        the filters (small imaging panels) or a median below 150 counts per cell in those genes, the calls are not
        reliable, so by default copy_number logs a warning and labels the cells `"undetermined"` without fitting the
        model. For such data, use :func:`~infercnvpy.tl.infercnv` with reference cells, smoothed with
        :func:`~infercnvpy.tl.smooth_cnv`.
    calculate_gene_values
        Also store the copy number log ratio of every gene (cells x genes) in
        `adata.layers["gene_values_{key_added}"]`, as :func:`~infercnvpy.tl.infercnv` does. Gene values come from the
        clones' segment means at gene resolution, not from the bins.
    bin_size
        Size of the genomic bins of the output, in bases.
    chunksize
        Number of cells processed at a time.
    random_state
        Seed for subsampling and k-means.
    key_added
        Name of the results in `adata.obs`, `adata.obsm` and `adata.uns`.
    inplace
        Write the results to `adata`. If `False`, return them.
    n_jobs
        Number of threads.

    Returns
    -------
    If `inplace=False`, a tuple `(obs, cna, uns, gene_values)`: a DataFrame with the `adata.obs` columns below, the
    values of `adata.obsm["X_copy_number"]` and `adata.uns["copy_number"]`, and the gene values (`None` unless
    `calculate_gene_values=True`). Otherwise `None`, and sets the following fields
    (named after `key_added`):

    `adata.obs["copy_number"]` : :class:`pandas.Categorical`
        `"aneuploid"` (posterior probability above 0.5), `"diploid"`, or `"not.defined"` for cells removed by the
        filters.
    `adata.obs["copy_number_score"]` : :class:`float`
        Posterior probability that the cell is aneuploid.
    `adata.obs["copy_number_llr"]` : :class:`float`
        Log likelihood ratio of the best aneuploid clone against the diploid clone (unscaled, without priors); a
        continuous per-cell score.
    `adata.obs["copy_number_clone"]` : :class:`pandas.Categorical`
        Clone of the cell: `"N0"` for the diploid clone, `"T1"`, `"T2"`, ... for aneuploid clones.
    `adata.obs["copy_number_spillover"]` : :class:`float`
        Fraction of the cell's counts that is best explained by the pooled expression of the aneuploid cells, mixed
        with its own type's diploid expression (0 to 1). High values in normal cell types point to segmentation merges
        or doublets with tumour cells; for cells of marker-rich (normal) types, up to 0.5 of this mixture is part of
        the diploid expectation, so that spillover alone does not make them aneuploid.
    `adata.obsm["X_copy_number"]` : :class:`numpy.ndarray`
        Copy number log ratios (float32), cells x genomic bins: the posterior mixture of the clones' segment means, a
        continuous score. `NaN` for cells removed by the filters.
    `adata.layers["gene_values_copy_number"]` : :class:`numpy.ndarray`
        Only if `calculate_gene_values=True`: log ratio per gene (float32), `NaN` for genes and cells removed by the
        filters.
    `adata.uns["copy_number"]["chr_pos"]` : :class:`dict`
        Maps each chromosome to the column of `X_copy_number` where it starts.
    `adata.uns["copy_number"]["bins"]` : :class:`pandas.DataFrame`
        `chromosome`, `start` and `end` of each bin.
    `adata.uns["copy_number"]["window_start"]`, `adata.uns["copy_number"]["window_end"]` : :class:`numpy.ndarray`
        Start and end of each bin, under the keys that :func:`~infercnvpy.tl.infercnv` uses for its windows.
    `adata.uns["copy_number"]["clones"]` : :class:`pandas.DataFrame`
        Per clone: `n_cells`, `aneuploid`, `amplitude` (smooth amplitude of its profile) and `burden` (fraction of
        genes in long non-neutral segments).
    `adata.uns["copy_number"]["clone_profiles"]` : :class:`pandas.DataFrame`
        Log ratio of each clone in each bin.
    `adata.uns["copy_number"]["segments"]` : :class:`pandas.DataFrame`
        Segments of the aneuploid clones: `clone`, `chromosome`, `start`, `end`, `n_genes`, `log_ratio` with a 95%
        interval (`ci_low`, `ci_high`), `state_ratio` (copy ratio of the HMM state) and `p_neutral` (mean posterior
        probability of the neutral state).
    `adata.uns["copy_number"]["genes"]` : :class:`pandas.DataFrame`
        The genes used, in genomic order, with their `dosage` weight (1: expression follows copy number fully).
    `adata.uns["copy_number"]["baseline"]` : :class:`str`
        How the first diploid cells were found.
    `adata.uns["copy_number"]["temperature"]` : :class:`float`
        Scale applied to the log likelihoods for the posterior.
    `adata.uns["copy_number"]["params"]` : :class:`dict`
        Parameters of this call.
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns):
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`."
        )
    params = _params_for_uns(
        layer=layer, reference_key=reference_key, reference_cat=reference_cat,
        exclude_chromosomes=exclude_chromosomes, exclude_genes=exclude_genes, min_genes=min_genes,
        min_detection=min_detection, window_size=window_size, n_clusters=n_clusters, segment_length=segment_length,
        min_burden=min_burden, normal_markers=normal_markers, n_iter=n_iter, max_fit_cells=max_fit_cells, bin_size=bin_size, random_state=random_state,
        **({} if connectivity_key is None else {"connectivity_key": connectivity_key, "n_hops": n_hops}),
    )  # fmt: skip
    if n_hops < 1:
        raise ValueError(f"`n_hops` must be at least 1, got {n_hops}.")
    if connectivity_key is not None:
        from infercnvpy.tl._spatial import _graph

        _graph(adata, connectivity_key)  # before the fit: fails early on a missing or distance graph
    rng = np.random.default_rng(random_state)
    expr = _as_matrix(adata.X if layer is None else adata.layers[layer], lazy=True)
    excluded = _excluded_genes(adata.var_names, exclude_genes)

    with numba_threads(n_jobs):
        source = _Rows(expr, chunksize)
        start = log.debug("copy_number: %d cells x %d genes", adata.n_obs, adata.n_vars)
        tm = _Timer()
        cells, lay, var, candidates = _filter_cells_genes(
            adata, source, excluded, exclude_chromosomes, min_genes, min_detection
        )
        tm("filter")
        n = len(cells)
        fit = (
            np.arange(n)
            if max_fit_cells is None or n <= max_fit_cells
            else np.sort(rng.choice(n, max_fit_cells, replace=False))
        )
        data = _Data(source, cells[fit], lay, window_size)
        n_fit = len(fit)
        tm("window counts")
        depth = float(np.median(data.size))
        if not force_calls and (lay.n_genes < _MIN_PANEL_GENES or depth < _MIN_MEDIAN_COUNTS):
            return _undetermined(adata, cells, lay, depth, bin_size, calculate_gene_values, key_added, params, inplace)
        hmm = _HMM(lay, segment_length)

        # expression clusters and their pooled profiles
        n_clusters = n_clusters or int(np.clip(n_fit // 50, 8, 40))
        pcs = data.expression_pcs(rng)
        tm("expression pca")
        clusters = _kmeans(pcs, min(n_clusters, n_fit), rng)
        tm("kmeans")
        Yc, Sc = data.pooled(clusters)
        tm("pooled")
        gene_var = _gene_variability(Yc, Sc, lay)
        burden = _pairwise_burden(hmm, Yc, Sc, gene_var, lay)
        groups = _groups(burden, min_burden)
        tm("pairwise hmm")

        gcell = groups[clusters]
        n_groups = int(groups.max()) + 1
        if reference_key is not None and reference_cat is not None:
            normal = np.isin(np.asarray(adata.obs[reference_key])[cells[fit]], _categories(reference_cat))
            if not normal.any():
                raise ValueError(f"No cell passing the filters has `{reference_key}` in {reference_cat!r}.")
            baseline_method = "reference"
        else:
            is_marker_var = np.asarray(adata.var_names.isin(_marker_list(normal_markers)))
            lay_seed, Yc_seed = _seed_genes(data, clusters, var, candidates, is_marker_var, min_detection)
            is_marker = is_marker_var[lay_seed.order]
            normal, baseline_method = _seed(Yc_seed, groups, gcell, lay_seed, is_marker)

        # clones start from the groups: those of the baseline cells form the diploid clone 0. After a pass, a
        # clearly flatter clone becomes diploid and the passes restart from it (not if markers chose the baseline).
        # After each pass, aneuploid clones whose profile is not autocorrelated along the genome merge into the
        # diploid clone, and the diploid clone defines the baseline
        em = _EM(data, hmm, gene_var, min_burden, clusters, Yc, Sc)
        if baseline_method != "reference" and normal_markers is not None:
            em.normal_type = _marker_types(Yc_seed, lay_seed, is_marker)
            log.debug("copy_number: marker-rich expression types %s", np.flatnonzero(em.normal_type))
        em.tm = tm
        frac = np.bincount(gcell, weights=normal, minlength=n_groups) / np.bincount(gcell, minlength=n_groups)
        lut = np.where(frac > 0.5, 0, 1 + np.arange(n_groups))
        assign = np.where(normal, 0, lut[gcell])
        fixed = baseline_method == "reference"
        reseeded = "markers" in baseline_method  # marker-rich groups outweigh flatness: no restart
        for pass_ in range(_N_PASSES):
            em.set_baseline(normal)
            assign, resp, mu, llr = em.run(assign, n_iter)
            k = int(assign.max()) + 1
            target = np.arange(k)
            if not fixed and not reseeded:
                Yk = data.pooled(assign, k, lay_seed)[0]
                sizes = np.bincount(assign, minlength=k)
                flat = np.array([_flatness(y, lay_seed) for y in Yk])
                small = sizes < min(_MIN_RESEED, int(sizes.max()))
                small[0] = False  # the diploid clone itself may be a rare normal population
                flat[small] = np.inf
                seed = int(np.argmin(flat))
                log.debug("copy_number: pass %d, clone flatness %s", pass_, np.round(flat, 3))
                if flat[seed] < flat[0] - _RESEED_GAP:
                    reseeded, baseline_method = True, "flattest clone"
                    target[0], target[seed] = seed, 0
                    assign = target[assign]
                    normal = assign == 0
                    continue
            amp = np.array([_smooth_amplitude(em.final[3][i], lay) for i in range(k)])
            log.debug("copy_number: pass %d, clone amplitudes %s", pass_, np.round(amp, 3))
            target[amp < _AMP_TUMOUR] = 0
            if (target == np.arange(k)).all():
                break
            assign = _compact(target[assign])
            if not fixed:
                normal = assign == 0
        else:
            assign, resp, mu, llr = em.run(assign, 1)
        dip = np.arange(len(mu)) == 0

        tm("em")
        if n_fit < n:  # assign every cell with the fitted model, in one pass over the counts
            pcs_all = data.project(source, cells)
            tm("project all cells")
            resp, llr = em.assign_all(source, cells, pcs_all, pcs, clusters)
            assign = np.argmax(resp, axis=1)
            tm("assign all cells")
        if connectivity_key is not None:
            resp, llr, linked = _smooth_over_graph(adata, connectivity_key, n_hops, cells, resp, llr)
            assign = np.where(linked, np.argmax(resp, axis=1), assign)
            tm("spatial smoothing")
        p_an = resp[:, ~dip].sum(axis=1)
        bins, chr_pos, M = lay.bins(np.arange(lay.n_genes), bin_size)
        cna = np.empty((adata.n_obs, len(bins)), dtype=np.float32)  # every row is written below
        cna[np.setdiff1d(np.arange(adata.n_obs), cells)] = np.nan
        names = np.array([("N" if d else "T") + str(i) for i, d in enumerate(dip)])
        _, pn_f, st_f, lr_f, v_f = em.final
        seg_values, segments = _segments(lr_f, v_f, st_f, pn_f, lay, names, hmm.means)
        segments = segments[~segments["clone"].isin(names[dip])].reset_index(drop=True)
        mu = seg_values
        gene_values = None
        if calculate_gene_values:
            gene_values = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32)
            gmu = seg_values.copy()
            gmu[dip] = 0.0
            gmu = gmu.astype(np.float32)
            K.mix_rows(resp.astype(np.float32), gmu, cells, lay.order, gene_values)
        mu[dip] = 0.0
        prof = np.ascontiguousarray((M @ mu.T).T, dtype=np.float32)
        K.mix_rows(resp.astype(np.float32), prof, cells, np.arange(prof.shape[1]), cna)
        tm("outputs")
        log.debug("copy_number: phase times (s) %s", {k: round(v, 2) for k, v in tm.times.items()})
        clones = _clone_table(em, assign, names, dip, lay)
        log.debug("copy_number: clones\n%s", clones.round(3).to_string())

    status = np.full(adata.n_obs, LABELS[2], dtype=object)
    status[cells] = np.where(p_an > 0.5, LABELS[0], LABELS[1])
    score = np.full(adata.n_obs, np.nan)
    score[cells] = p_an
    llr_all = np.full(adata.n_obs, np.nan)
    llr_all[cells] = llr
    clone = np.full(adata.n_obs, None, dtype=object)
    clone[cells] = names[assign]
    spill = np.full(adata.n_obs, np.nan)
    spill[cells] = em.spill
    obs = pd.DataFrame(
        {
            key_added: pd.Categorical(status, categories=LABELS),
            f"{key_added}_score": score,
            f"{key_added}_llr": llr_all,
            f"{key_added}_clone": pd.Categorical(clone),
            f"{key_added}_spillover": spill,
        },
        index=adata.obs_names,
    )
    uns = {
        "chr_pos": chr_pos,
        "bins": bins,
        "window_start": bins["start"].to_numpy(),  # as written by tl.infercnv, for region selection
        "window_end": bins["end"].to_numpy(),
        "clones": clones,
        "clone_profiles": pd.DataFrame((M @ mu.T).T, index=names),
        "segments": segments,
        "genes": pd.DataFrame(
            {"chromosome": lay.chrom, "start": lay.start, "end": lay.end, "dosage": em.dosage},
            index=np.asarray(adata.var_names)[lay.order],
        ),
        "baseline": baseline_method,
        "temperature": em.tau,
        "params": params,
    }
    n_an = int((status == LABELS[0]).sum())
    log_result(
        "copy_number",
        "%d aneuploid and %d diploid cells in %d clones; %d of %d cells not defined",
        *(n_an, n - n_an, len(mu), adata.n_obs - n, adata.n_obs),
        dropped=adata.n_obs - n,
        total=adata.n_obs,
        remedy="lower `min_genes`",
        time=start,
    )
    if not inplace:
        return obs, cna, uns, gene_values
    for c in obs.columns:
        adata.obs[c] = obs[c]
    adata.obsm[f"X_{key_added}"] = cna
    if gene_values is not None:
        adata.layers[f"gene_values_{key_added}"] = gene_values
    adata.uns[key_added] = uns
    return None


def _undetermined(adata, cells, lay, depth, bin_size, calculate_gene_values, key_added, params, inplace):
    """The outputs of copy_number with every filtered cell `"undetermined"` (too little copy number information)."""
    log.warning(
        "copy_number: %d genes pass the filters, with a median of %d counts per cell; reliable calls need at least "
        "%d genes and %d counts per cell, so %d cells are labelled %r. For small panels, use `tl.infercnv` with "
        "reference cells, smoothed with `tl.smooth_cnv`; `force_calls=True` calls the cells anyway.",
        lay.n_genes, depth, _MIN_PANEL_GENES, _MIN_MEDIAN_COUNTS, len(cells), UNDETERMINED,
    )  # fmt: skip
    bins, chr_pos, _ = lay.bins(np.arange(lay.n_genes), bin_size)
    status = np.full(adata.n_obs, LABELS[2], dtype=object)
    status[cells] = UNDETERMINED
    obs = pd.DataFrame(
        {
            key_added: pd.Categorical(status, categories=[*LABELS, UNDETERMINED]),
            f"{key_added}_score": np.nan,
            f"{key_added}_llr": np.nan,
            f"{key_added}_clone": pd.Categorical(np.full(adata.n_obs, None, dtype=object)),
            f"{key_added}_spillover": np.nan,
        },
        index=adata.obs_names,
    )
    cna = np.full((adata.n_obs, len(bins)), np.nan, dtype=np.float32)
    gene_values = np.full((adata.n_obs, adata.n_vars), np.nan, dtype=np.float32) if calculate_gene_values else None
    uns = {
        "chr_pos": chr_pos,
        "bins": bins,
        "window_start": bins["start"].to_numpy(),
        "window_end": bins["end"].to_numpy(),
        "undetermined": {"n_genes": lay.n_genes, "median_counts": depth},
        "params": params,
    }
    if not inplace:
        return obs, cna, uns, gene_values
    for c in obs.columns:
        adata.obs[c] = obs[c]
    adata.obsm[f"X_{key_added}"] = cna
    if gene_values is not None:
        adata.layers[f"gene_values_{key_added}"] = gene_values
    adata.uns[key_added] = uns
    return None


def _smooth_over_graph(adata, connectivity_key, n_hops, cells, resp, llr):
    """Posteriors and llr averaged over each cell and its neighbours in `obsp[connectivity_key]`.

    Uses the kernel of `tl.smooth_cnv`, on the graph between the cells that pass the filters. Also returns which cells
    have neighbours (the others keep their values).
    """
    from infercnvpy.tl._spatial import _graph, _hop_dense

    G = _graph(adata, connectivity_key)
    if G.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(f"`adata.obsp[{connectivity_key!r}]` has shape {G.shape}, expected {(adata.n_obs,) * 2}.")
    G = G[cells][:, cells].tocsr()
    graph = (G.indptr, G.indices, G.data.astype(np.float64, copy=False), -1.0)
    X = np.column_stack([resp, llr]).astype(np.float64)
    out = np.empty_like(X)
    for _ in range(n_hops):
        _hop_dense(*graph, X, out)
        X, out = out, X
    return X[:, :-1].astype(resp.dtype), X[:, -1].astype(np.float64), np.diff(G.indptr) > 0


def _row_blocks(source: _Rows, rows: np.ndarray):
    """Yield `(i0, i1, X, rws)` for each block of the matrix that holds any of `rows` (sorted):

    `rows[i0:i1]` are the rows `rws` of the CSR block `X`. Blocks are read whole (in memory: no copy), the kernels
    pick the rows.
    """
    i0 = 0
    for X, rws in source.rows_of(rows):
        yield i0, i0 + len(rws), _csr(X), rws.astype(np.int64)
        i0 += len(rws)


def _filter_cells_genes(adata, source, excluded, exclude_chromosomes, min_genes, min_detection):
    """Cells and genes that pass the filters, in one pass over the counts.

    Cells have at least `min_genes` detected genes; genes are detected in at least `min_detection` of these cells.
    Genes count if they are placed (chromosome and start) on a chromosome named like "chr1" and not excluded; these
    candidates and the gene positions (a missing end is the start, as in `tl.infercnv`) are returned too.
    """
    var = pd.DataFrame({c: np.asarray(adata.var[c]) for c in ("chromosome", "start", "end")}, index=adata.var_names)
    var["end"] = var["end"].fillna(var["start"])
    chrom = var["chromosome"].astype(object)
    named = chrom.notna() & var["start"].notna() & chrom.astype(str).str.startswith("chr") & (chrom != "chrM")
    if not named.any():
        raise ValueError(
            'None of the chromosome names in `adata.var["chromosome"]` start with "chr". '
            'infercnvpy only uses chromosomes named like "chr1".'
        )
    if exclude_chromosomes is not None:
        named &= ~chrom.isin(_as_list(exclude_chromosomes))
    candidates = named.to_numpy() & ~excluded
    det = np.zeros((min(numba.get_num_threads(), 16), adata.n_vars), dtype=np.int64)
    ndet = []
    for X, rws in source.rows_of(None):  # all rows: each block's rows are contiguous
        if len(rws):
            X = _csr(X)
            nd = np.empty(len(rws), dtype=np.int64)
            K.row_stats(X.data, X.indices, X.indptr, int(rws[0]), min_genes, nd, det)
            ndet.append(nd)
    cells = np.flatnonzero(np.concatenate(ndet) >= min_genes)
    if len(cells) < 2:
        raise ValueError(f"Fewer than 2 cells have at least {min_genes} detected genes.")
    used = (det.sum(axis=0) / len(cells) >= min_detection) & candidates
    if not used.any():
        raise ValueError("No gene passes the filters.")
    lay = _Layout(var, used)
    log.debug("copy_number: %d of %d cells and %d genes pass the filters", len(cells), adata.n_obs, lay.n_genes)
    return cells, lay, var, candidates


def _seed_genes(data, clusters, var, candidates, is_marker, min_detection):
    """Layout and pooled cluster counts of the genes that the baseline seed scores.

    These are the genes detected in at least `min_detection` of the cells of any expression cluster, and the marker
    genes. The model's genes are detected in `min_detection` of all cells: where a tumour is the majority, its losses
    fall below that threshold more often than its gains, and the genes left tilt the genomic flatness of every group
    towards the tumour. Per cluster, a gene lost in the tumour is still detected in the normal cells.
    """
    lay_c = _Layout(var, candidates)
    Yc, _, det = data.pooled(clusters, lay=lay_c, detected=True)
    rate = det / np.bincount(clusters)[:, None]
    keep = (rate >= min_detection).any(axis=0) | is_marker[lay_c.order]
    used = np.zeros(len(var), dtype=bool)
    used[lay_c.order[keep]] = True
    return _Layout(var, used), Yc[:, keep]


def _as_list(x) -> list:
    """A string as a list of one, any other collection as a list."""
    return [x] if isinstance(x, str) else list(x)


class _Timer:
    """Wall time per phase (debug log)."""

    def __init__(self):
        import time

        self._clock = time.perf_counter
        self.t = self._clock()
        self.times: dict[str, float] = {}

    def __call__(self, name: str):
        t = self._clock()
        self.times[name] = self.times.get(name, 0.0) + t - self.t
        self.t = t


class _Data:
    """Access to the counts of the cells passing the filters: windows of genes per cell, pooled gene counts."""

    def __init__(self, source: _Rows, cells: np.ndarray, lay, window_size: int):
        self.source, self.cells, self.lay = source, cells, lay
        bounds, win_chrom = [], []
        for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True)):
            nw = max(1, g // window_size)
            bounds.extend(g0 + np.arange(nw) * window_size)
            win_chrom.extend([c] * nw)
        self.wbounds = np.array([*bounds, lay.n_genes], dtype=np.int64)
        self.gene_win = np.repeat(np.arange(len(bounds)), np.diff(self.wbounds))
        self.win_parity = np.arange(len(bounds)) % 2
        n_win = len(self.wbounds) - 1
        Yw = []
        for i0, i1, X, rws in _row_blocks(source, cells):
            out = np.empty((i1 - i0, n_win), dtype=np.float32)
            K.window_counts(
                X.data, X.indices, X.indptr, rws, lay.col_map, self.gene_win, out, n_kernel_chunks(len(out))
            )
            Yw.append(out)
        self.Yw = np.concatenate(Yw)
        self.size = self.Yw.sum(axis=1, dtype=np.float64)

    def blocks(self, rows: np.ndarray | None = None):
        """Yield `(i0, i1, X)`: positions `i0:i1` in `rows` (default: all cells) and their counts (CSR, genomic order)."""
        rows = self.cells if rows is None else rows
        i0 = 0
        for X, rws in self.source.rows_of(rows):
            yield i0, i0 + len(rws), _csr(X)[rws][:, self.lay.order].astype(np.float64)
            i0 += len(rws)

    def pooled(self, labels: np.ndarray, k: int | None = None, lay=None, detected: bool = False):
        """Pooled gene counts (k x genes of `lay`, default: the model's) and pooled size factors of each label.

        With `detected`, also the number of cells of each label in which each gene is detected.
        """
        lay = self.lay if lay is None else lay
        k = int(labels.max()) + 1 if k is None else k
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        slabs = np.zeros((min(numba.get_num_threads(), 16), k, lay.n_genes))
        det = np.zeros_like(slabs) if detected else np.zeros((0, 0, 0))
        for i0, i1, X, rws in _row_blocks(self.source, self.cells):
            K.pooled_counts(X.data, X.indices, X.indptr, rws, lay.col_map, labels[i0:i1], slabs, det)
        out = slabs.sum(axis=0), np.bincount(labels, weights=self.size, minlength=k)
        return (*out, det.sum(axis=0)) if detected else out

    def expression_pcs(self, rng) -> np.ndarray:
        """First principal components of log-normalised expression of the most variable genes."""
        sub = _subsample(np.arange(len(self.cells)), _N_SUB, rng)
        Xs = sp.vstack([Xb for _, _, Xb in self.blocks(self.cells[sub])]).tocsr()
        Xs = sp.diags(1e4 / np.maximum(self.size[sub], 1)) @ Xs
        Xs.data = np.log1p(Xs.data)
        mean = np.asarray(Xs.mean(axis=0)).ravel()
        var = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel() - mean**2
        hvg = np.sort(np.argsort(var)[::-1][: min(_N_HVG, len(var))])
        D = Xs[:, hvg].toarray() - mean[hvg]
        n_pcs = max(1, min(_N_PCS, D.shape[0] - 1, D.shape[1]))
        proj = _top_right_singular_vectors(D, n_pcs)
        self.hvg_of_gene = np.full(self.lay.n_genes, -1, dtype=np.int64)
        self.hvg_of_gene[hvg] = np.arange(len(hvg))
        self.loadings = np.ascontiguousarray(proj)
        self.pc_offset = mean[hvg] @ proj
        return self.project(self.source, self.cells)

    def project(self, source, rows) -> np.ndarray:
        """Principal component coordinates of the cells `rows` (any cells, not only those of this object)."""
        out = []
        for i0, i1, X, rws in _row_blocks(source, rows):
            o = np.empty((i1 - i0, self.loadings.shape[1]))
            K.project_rows(
                X.data, X.indices, X.indptr, rws, self.lay.col_map, self.hvg_of_gene, self.loadings, o,
                n_kernel_chunks(i1 - i0),
            )  # fmt: skip
            out.append(o - self.pc_offset)
        return np.concatenate(out)


def _top_right_singular_vectors(D: np.ndarray, k: int) -> np.ndarray:
    """Top `k` right singular vectors of `D` (columns).

    An exact SVD for up to `_EXACT_SVD` rows, else the eigenvectors of `D.T @ D` (exact too, and fast for many rows
    and few columns).
    """
    if D.shape[0] <= _EXACT_SVD:
        return np.linalg.svd(D, full_matrices=False)[2][:k].T
    w, vec = np.linalg.eigh(D.T @ D)
    return vec[:, np.argsort(w)[::-1][:k]]


class _HMM:
    """Copy number HMM over the genes in genomic order, with transitions by genomic distance."""

    def __init__(self, lay, segment_length: float):
        centre = (lay.start + lay.end) / 2.0
        d = np.diff(centre, prepend=centre[0])
        stay = np.exp(-np.maximum(d, 0) / segment_length)
        stay[lay.g0s] = 0.0
        self.stay = stay
        self.means = np.log(np.array(STATE_RATIOS))
        prior = np.full(len(STATE_RATIOS), (1 - _P_NEUTRAL) / (len(STATE_RATIOS) - 1))
        prior[np.argmin(np.abs(self.means))] = _P_NEUTRAL
        self.means[np.argmin(np.abs(self.means))] = 0.0
        self.log_prior = np.log(prior)

    def __call__(self, y: np.ndarray, v: np.ndarray):
        y = np.ascontiguousarray(np.atleast_2d(y), dtype=np.float64)
        v = np.ascontiguousarray(np.atleast_2d(v), dtype=np.float64)
        P, G = y.shape
        pm, pn = np.empty((P, G)), np.empty((P, G))
        st, ll = np.empty((P, G), dtype=np.int64), np.empty(P)
        K.hmm_posteriors(y, v, self.stay, self.means, self.log_prior, _NU, pm, pn, st, ll)
        return pm, pn, st, ll


def _log_ratio(Y, E, Yref=None):
    """Gene-level log ratio of pooled counts `Y` to their expectation `E`, and its Poisson variance."""
    lr = np.log((Y + 0.5) / (E + 0.5))
    var = 1.0 / (Y + 0.5)
    if Yref is not None:
        var = var + 1.0 / (Yref + 0.5)
    return lr, var


def _remove_level(lr: np.ndarray, v: np.ndarray, b: np.ndarray, lay) -> np.ndarray:
    """`lr` (profiles x genes) minus its weighted regression on gene covariates that are not copy number.

    The covariates are the gene's baseline log expression and log length, and their squares. Cells and clusters
    differ in how their counts spread over highly and lowly expressed genes (capture and saturation), and nuclei
    against whole cells in how they spread over long and short genes (intronic reads); both show up along gene-dense
    chromosomes such as chr19 and are not copy number.
    """
    z = np.log(b)
    z = (z - z.mean()) / z.std()
    cols = [np.ones_like(z), z, z**2 - 1]
    ln = np.log(np.maximum(lay.end - lay.start, 1).astype(np.float64))
    if ln.std() > 0:
        ln = (ln - ln.mean()) / ln.std()
        cols += [ln, ln**2 - 1]
    Z = np.stack(cols, axis=1)
    out = np.empty_like(lr)
    for i in range(len(lr)):
        w = 1.0 / v[i]
        r = np.clip(lr[i], -3, 3)
        beta = np.linalg.lstsq(Z.T @ (Z * w[:, None]), Z.T @ (w * r), rcond=None)[0]
        out[i] = lr[i] - Z[:, 1:] @ beta[1:]
    return out


def _expression_deviation(resid: np.ndarray, pois: np.ndarray) -> np.ndarray:
    """Gene-level expression deviation of each clone: what the smooth copy number profile leaves.

    It is shrunk towards 0 by its Poisson noise, and carries the clone's cell type, without any copy number signal.
    """
    out = np.empty_like(resid)
    for i in range(len(resid)):
        r = np.clip(resid[i], -3, 3)
        tau2 = max(float(np.median(r**2)) / 0.455 - float(np.median(pois[i])), 1e-3)
        out[i] = r * tau2 / (tau2 + pois[i])
    return out


def _robust_var(x: np.ndarray) -> float:
    d = np.diff(x)
    return float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2 / 2)


def _scale(lr, pois, gene_var):
    """Total variance per gene: expression variance (`gene_var` scaled to the profile) plus Poisson variance."""
    resid = lr / np.sqrt(gene_var + pois)
    s2 = max(_robust_var(resid), 1e-4)
    return s2 * (gene_var + pois)


def _gene_variability(Yc, Sc, lay) -> np.ndarray:
    """Relative expression variability of each gene across the clusters, after removing smooth trends."""
    L = np.log((Yc + 0.5) / Sc[:, None])
    L -= np.average(L, axis=0, weights=Sc)
    from scipy.ndimage import median_filter

    R = np.empty_like(L)
    for g0, g in zip(lay.g0s, lay.gs, strict=True):
        R[:, g0 : g0 + g] = L[:, g0 : g0 + g] - median_filter(L[:, g0 : g0 + g], size=(1, 51), mode="nearest")
    pois = (1.0 / (Yc + 0.5)).mean(axis=0)
    v = np.maximum(np.average(R**2, axis=0, weights=Sc) - pois, 0)
    v = v + np.median(pois)
    return np.maximum(v / np.median(v), 0.1)


def _pairwise_burden(hmm, Yc, Sc, gene_var, lay) -> np.ndarray:
    """Fraction of genes in non-neutral HMM states of the difference of each pair of cluster profiles."""
    k = len(Yc)
    L = np.log((Yc + 0.5) / Sc[:, None])
    P = 1.0 / (Yc + 0.5)
    bmean = (Yc.sum(0) + 0.5) / (Yc.sum() + 0.5 * Yc.shape[1])
    burden = np.zeros((k, k))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for s in range(0, len(pairs), 200):
        batch = pairs[s : s + 200]
        y = np.array([L[i] - L[j] for i, j in batch])
        y -= np.median(y, axis=1, keepdims=True)
        v = np.array([_scale(y[t], P[i] + P[j], gene_var) for t, (i, j) in enumerate(batch)])
        y = _remove_level(y, v, bmean, lay)
        pm, pn, _, _ = hmm(y, v)
        for t, (i, j) in enumerate(batch):
            burden[i, j] = burden[j, i] = _burden(pm[t], pn[t], lay)
    return burden


def _marker_list(markers) -> list[str]:
    if markers is None:
        return []
    if isinstance(markers, Mapping):
        return [g for genes in markers.values() for g in genes]
    return list(markers)


def _seed(Yc, groups, gcell, lay, is_marker) -> tuple[np.ndarray, str]:
    """Cells of the first baseline: the groups in the flat cluster of a 2-means split of the group flatness (if the split is clear, else the flattest group), plus groups that express immune, endothelial or stromal markers far above the marker-poor groups and are flatter than the median group.

    Cells of the first baseline: the groups in the flat cluster of a 2-means split of the group flatness (if the
    split is clear, else the flattest group), plus groups that express immune, endothelial or stromal markers far
    above the marker-poor groups and are flatter than the median group.
    """
    n_groups = int(groups.max()) + 1
    sizes = np.bincount(gcell, minlength=n_groups)
    Yg = np.array([Yc[groups == g].sum(0) for g in range(n_groups)])
    flat = np.array([_flatness(y, lay) for y in Yg])
    ok = sizes >= min(_MIN_SEED, int(sizes.max()))
    seed = np.zeros(n_groups, dtype=bool)
    seed[np.flatnonzero(ok)[np.argmin(flat[ok])]] = True
    method = "flattest group"
    lo, hi = _two_means(flat[ok], sizes[ok])
    if hi - lo > _FLAT_GAP:
        seed |= ok & (flat < (lo + hi) / 2)
        method = "flat groups"
    mfrac = Yg[:, is_marker].sum(1) / Yg.sum(1)
    # lower quartile, not median: in samples where normal cells are the majority the median group is itself marker-rich
    low = np.quantile(mfrac[ok], 0.25)
    marked = ok & (mfrac > np.maximum(_MARKER_FOLD * low, _MARKER_MIN)) & (flat < np.median(flat[ok]))
    if marked.any():
        seed |= marked
        method += " + markers"
    log.debug("copy_number: group flatness %s, marker fraction (per mille) %s, sizes %s, seed %s",
              np.round(flat, 3), np.round(1000 * mfrac, 2), sizes, np.flatnonzero(seed))  # fmt: skip
    return seed[gcell], method


def _marker_types(Yc, lay, is_marker) -> np.ndarray:
    """Expression types that are marker-rich (same rule as the marker-rich seed groups) and flatter than the median."""
    mfrac = Yc[:, is_marker].sum(1) / Yc.sum(1)
    flat = np.array([_flatness(y, lay) for y in Yc])
    low = np.quantile(mfrac, 0.25)
    return (mfrac > max(_MARKER_FOLD * low, _MARKER_MIN)) & (flat < np.median(flat))


def _two_means(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Means of the two clusters of the weighted 1-D 2-means split of `x` (the best threshold between sorted values)."""
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    o = np.argsort(x)
    x, w = x[o], w[o].astype(np.float64)
    best, out = np.inf, (float(x[0]), float(x[-1]))
    for i in range(1, len(x)):
        m1, m2 = np.average(x[:i], weights=w[:i]), np.average(x[i:], weights=w[i:])
        cost = (w[:i] * (x[:i] - m1) ** 2).sum() + (w[i:] * (x[i:] - m2) ** 2).sum()
        if cost < best:
            best, out = cost, (float(m1), float(m2))
    return out


def _burden(pm: np.ndarray, pn: np.ndarray, lay) -> float:
    """Fraction of genes in runs of at least `_MIN_SEG_GENES` genes in a non-neutral state with a posterior mean log ratio of at least `_MIN_EFFECT`.

    Fraction of genes in runs of at least `_MIN_SEG_GENES` genes in a non-neutral state with a posterior mean log
    ratio of at least `_MIN_EFFECT`. Short runs are mostly clustered gene families of a cell type. On small panels
    (a typical chromosome with fewer than `4 _MIN_SEG_GENES` genes) the shortest run is a quarter of its genes (at
    least 4).
    """
    g = int(np.median(lay.gs))
    min_genes = _MIN_SEG_GENES if g >= 4 * _MIN_SEG_GENES else max(4, g // 4)
    m = np.concatenate([[False], (pn < 0.5) & (np.abs(pm) >= _MIN_EFFECT), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    lengths = edges[1::2] - edges[::2]
    return float(lengths[lengths >= min_genes].sum() / len(pm))


def _groups(burden: np.ndarray, cut: float) -> np.ndarray:
    """Groups of clusters (complete linkage on the burden) whose pairwise burdens are all below `cut`."""
    if len(burden) < 2:
        return np.zeros(len(burden), dtype=np.int64)
    Z = hierarchy.linkage(squareform(burden, checks=False), method="complete")
    return hierarchy.fcluster(Z, cut, criterion="distance").astype(np.int64) - 1


def _flatness(Y: np.ndarray, lay) -> float:
    """Lag-1 autocorrelation along the genome of the window medians of a pooled log profile (not centred per gene)."""
    lr = np.log(Y + 0.5)
    size = _block_size(lay)
    vals, chrom = [], []
    for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True)):
        nw = max(1, g // size)
        for w in range(nw):
            e = g0 + g if w == nw - 1 else g0 + (w + 1) * size
            vals.append(np.median(lr[g0 + w * size : e]))
            chrom.append(c)
    x = np.array(vals) - np.mean(vals)
    same = np.diff(chrom) == 0
    return float((x[1:] * x[:-1])[same].sum() / max((x**2).sum(), 1e-12))


def _segments(lr: np.ndarray, v: np.ndarray, st: np.ndarray, pn: np.ndarray, lay, names, means):
    """Segments of each profile and the gene values they give.

    A segment is a run of genes with the same most probable HMM state on one chromosome. Its value is the
    precision-weighted mean of the (clipped) gene log ratios, with a 95% interval from the summed precisions (the
    variances are scaled to the robust spread of the profile, see `_scale`). Returns the gene values (profiles x
    genes) and a table with one row per segment.
    """
    out = np.empty_like(lr)
    starts = np.zeros(lr.shape[1], dtype=bool)
    starts[lay.g0s] = True
    frames = []
    for i in range(len(lr)):
        b = starts.copy()
        b[1:] |= st[i, 1:] != st[i, :-1]
        seg = np.cumsum(b) - 1
        w = 1.0 / v[i]
        den = np.bincount(seg, weights=w)
        mean = np.bincount(seg, weights=w * np.clip(lr[i], -2, 2)) / den
        out[i] = mean[seg]
        first = np.flatnonzero(b)
        last = np.r_[first[1:], len(seg)] - 1
        half = 1.96 / np.sqrt(den)
        frames.append(
            pd.DataFrame(
                {
                    "clone": names[i],
                    "chromosome": lay.chrom[first],
                    "start": lay.start[first],
                    "end": lay.end[last],
                    "n_genes": last - first + 1,
                    "log_ratio": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "state_ratio": np.exp(means[st[i, first]]),
                    "p_neutral": np.bincount(seg, weights=pn[i]) / np.bincount(seg),
                }
            )
        )
    return out, pd.concat(frames, ignore_index=True)


def _clone_table(em, assign, names, dip, lay) -> pd.DataFrame:
    """One row per clone: its number of cells, whether it is aneuploid, its smooth amplitude and burden."""
    mu, pn, _, lr, _ = em.final
    return pd.DataFrame(
        {
            "n_cells": np.bincount(assign, minlength=len(mu)),
            "aneuploid": ~dip,
            "amplitude": [_smooth_amplitude(x, lay) for x in lr],
            "burden": [0.0 if d else _burden(mu[i], pn[i], lay) for i, d in enumerate(dip)],
        },
        index=names,
    )


class _EM:
    """Copy number clones of cells with known expression types.

    The expected counts of a cell are its size factor times the baseline (pooled diploid cells) times the
    expression deviation of its type (an expression cluster: the gene-level part of its pooled profile that the HMM
    does not explain) times the copy ratio of its clone. Clone 0 is diploid (ratio 1); the profile of every other
    clone is the HMM on the gene-level log ratio of its pooled counts to their expectation. Cells are assigned by the
    negative binomial likelihood of their counts in windows of genes. Profiles are cross-fitted: windows of one
    parity are scored with the profile fitted on the genes of the other parity, so that the noise that selected a
    clone's cells does not shape the profile they are scored with.
    """

    def __init__(self, data: _Data, hmm: _HMM, gene_var, min_burden, types, Yt, St):
        self.data, self.hmm, self.gene_var, self.min_burden = data, hmm, gene_var, min_burden
        self.types, self.Yt, self.St = np.ascontiguousarray(types, dtype=np.int64), Yt, St
        self.n_types = len(St)
        self.calibrate = True
        self.normal_type = np.zeros(len(St), dtype=bool)  # types whose diploid expectation keeps their smooth part
        self.tm = _Timer()
        self.dosage = np.ones(Yt.shape[1])
        self.neutral = int(np.argmin(np.abs(hmm.means)))
        par = data.win_parity[data.gene_win]
        self.masks = (par == 0, par == 1)

    def set_baseline(self, normal):
        """Baseline from the cells `normal`, and the expression deviations of the types relative to it."""
        self.Yr = self.data.pooled(normal.astype(np.int64), 2)[0][1]
        self.b = (self.Yr + 0.5) / (self.Yr + 0.5).sum()
        lr, pois = _log_ratio(self.Yt, self.St[:, None] * self.b[None, :], self.Yr[None, :])
        v = np.array([_scale(lr[i], pois[i], self.gene_var) for i in range(len(lr))])
        lr = _remove_level(lr, v, self.b, self.data.lay)
        pm, _, _, _ = self.hmm(lr, v)
        self.e = _expression_deviation(lr - pm, pois)
        self.type_smooth = pm
        lay = self.data.lay
        self.type_blocks = np.array([_block_medians(x - _mode(x, lay), lay)[0] for x in lr])

    def type_priors(self, mu):
        """Prior clone probabilities of the cells of each type (types x clones).

        The pooled profile of a type, in blocks of 25 genes, is fitted as a non-negative mixture of the aneuploid clone
        profiles (weights summing to at most 1, the rest is diploid). A type of normal cells has weights near 0 even
        when its expression differs from the baseline smoothly, a tumour type near 1, and a mixed type its tumour
        fraction. The fit uses the pooled profiles, not the current assignments, so it does not reinforce them.
        """
        from scipy.optimize import nnls

        k = len(mu)
        pri = np.zeros((self.n_types, k))
        pri[:, 0] = 1.0
        if k > 1:
            lay = self.data.lay
            Mb = np.array([_block_medians(m, lay)[0] for m in mu[1:]]).T
            for t in range(self.n_types):
                w = nnls(Mb, self.type_blocks[t])[0]
                w = w / max(1.0, w.sum())
                pri[t, 1:] = w
                pri[t, 0] = 1.0 - w.sum()
        pri = np.maximum(pri, _MIN_PRIOR)
        return pri / pri.sum(axis=1, keepdims=True)

    def profiles(self, assign, k):
        """Gene-level log ratios of the clones against their expectation (with the deviations of their cells' types)."""
        data = self.data
        self.tm("em other")
        Y, _ = data.pooled(assign, k)
        self.tm("em pooled")
        Skt = np.zeros((k, self.n_types))
        np.add.at(Skt, (assign, self.types), data.size)
        E = self.b[None, :] * (Skt @ np.exp(self.e))
        lr, pois = _log_ratio(Y, E, self.Yr[None, :])
        v = np.array([_scale(lr[i], pois[i], self.gene_var) for i in range(k)])
        lr = _remove_level(lr, v, self.b, self.data.lay)
        return lr - np.array([_mode(x, self.data.lay) for x in lr])[:, None], v

    def fit(self, lr, v, cross: bool):
        """HMM posterior mean, P(neutral) and states per clone; with `cross`.

        each gene's value comes from the fit on the genes of the other window parity.
        """
        if not cross:
            mu, pn, st, _ = self.hmm(lr, v)
        else:
            mu, pn, st = np.empty_like(lr), np.empty_like(lr), np.empty(lr.shape, dtype=np.int64)
            for m in self.masks:
                f_mu, f_pn, f_st, _ = self.hmm(lr, np.where(m[None, :], v, 1e6))
                mu[:, ~m], pn[:, ~m], st[:, ~m] = f_mu[:, ~m], f_pn[:, ~m], f_st[:, ~m]
        mu[0], pn[0], st[0] = 0.0, 1.0, self.neutral
        self.tm("em hmm")
        return mu, pn, st

    def loglik(self, mu, assign, windows=None, report: bool = False):
        """Negative binomial log likelihood of each cell under each clone (cells x clones), on `windows` (mask)."""
        data = self.data
        wb = data.wbounds[:-1]
        share = np.empty((self.n_types, len(mu), len(wb)))
        for t in range(self.n_types):
            expo = self.dosage[None, :] * mu + self.e[t][None, :]
            if self.normal_type[t]:  # marker-rich normal type: its diploid expectation keeps its smooth deviation
                expo[0] = expo[0] + self.type_smooth[t]
            share[t] = np.add.reduceat(self.b[None, :] * np.exp(expo), wb, axis=1)
        use = np.ones(len(wb), dtype=np.bool_) if windows is None else np.asarray(windows, dtype=np.bool_)
        share = share * use
        share = np.ascontiguousarray(share / share.sum(axis=2, keepdims=True))
        size = data.size if windows is None else data.Yw @ use.astype(data.Yw.dtype)
        size = np.maximum(size.astype(np.float64), 1e-3)
        n_chunks = min(numba.get_num_threads(), 16)
        num, den = np.zeros((n_chunks, len(wb))), np.zeros((n_chunks, len(wb)))
        assign = np.ascontiguousarray(assign, dtype=np.int64)
        K.nb_moments(data.Yw, size, self.types, assign, share, use, num, den)
        inv_phi = np.clip(num.sum(0) / np.maximum(den.sum(0), 1e-12), 1e-3, 1e2)
        # spillover: the pooled window shares of the aneuploid cells, mixed into the diploid expectation of cells of
        # marker-rich types (segmentation merges with tumour cells in imaging data)
        tumour = (assign > 0).astype(np.float64)
        g = (tumour @ data.Yw) * use if tumour.any() else np.zeros(len(wb))
        g = np.ascontiguousarray(g / max(g.sum(), 1e-12))
        spill_ok = self.normal_type & bool(tumour.any())
        ll = np.empty((len(assign), len(mu)))
        self.spill = np.empty(len(assign))
        K.nb_loglik_types(
            data.Yw, size, self.types, share, 1.0 / inv_phi, use, g, spill_ok, _MAX_SPILL, report, ll, self.spill
        )
        self.tm("em loglik")
        self.model = (share, 1.0 / inv_phi, use, g, spill_ok)
        return ll

    def assign_all(self, source, cells, pcs_all, pcs_fit, types_fit):
        """Posterior clone probabilities and log likelihood ratios of all `cells` under the fitted model.

        A cell's type is the nearest k-means centre (fitted cells) in principal component space; its likelihood comes
        from one fused pass over its counts (`assign_loglik`), with the temperature and type priors of the fit.
        """
        n_types = self.n_types
        centres = np.zeros((n_types, pcs_fit.shape[1]))
        np.add.at(centres, types_fit, pcs_fit)
        centres /= np.maximum(np.bincount(types_fit, minlength=n_types), 1)[:, None]
        c2 = (centres**2).sum(axis=1)
        types = np.concatenate(
            [
                np.argmin(c2[None, :] - 2 * pcs_all[i : i + 20000] @ centres.T, axis=1)
                for i in range(0, len(pcs_all), 20000)
            ]
        ).astype(np.int64)
        share, phi, use, g, spill_ok = self.model
        k = share.shape[1]
        ll = np.empty((len(cells), k))
        self.spill = np.empty(len(cells))
        for i0, i1, X, rws in _row_blocks(source, cells):
            K.assign_loglik(
                X.data, X.indices, X.indptr, rws, self.data.lay.col_map, self.data.gene_win, types[i0:i1],
                share, phi, use, g, spill_ok, _MAX_SPILL, True, ll[i0:i1], np.empty(i1 - i0), self.spill[i0:i1],
                n_kernel_chunks(i1 - i0),
            )  # fmt: skip
        llr = ll[:, 1:].max(axis=1) - ll[:, 0] if k > 1 else np.zeros(len(ll))
        ll = self.tau * ll + np.log(self.priors)[types]
        resp = np.exp(ll - ll.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)
        return resp, llr

    def temperature(self, mu, assign) -> float:
        """Scale of the log likelihoods that calibrates them: the slope of a logistic regression (no intercept) of the sign of each cell's aneuploid-vs-diploid log likelihood ratio on one half of the windows on the ratio from the other half.

        Scale of the log likelihoods that calibrates them: the slope of a logistic regression (no intercept) of
        the sign of each cell's aneuploid-vs-diploid log likelihood ratio on one half of the windows on the
        ratio from the other half. The window likelihoods are not independent (expression programs, dispersion
        misfit), so the raw ratios are overconfident.
        """
        if len(mu) < 2:
            return 1.0
        par = self.data.win_parity
        half = [self.loglik(mu, assign, par == p) for p in (0, 1)]
        r = [h[:, 1:].max(axis=1) - h[:, 0] for h in half]
        x = np.concatenate([r[0], r[1]])
        y = np.concatenate([r[1] > 0, r[0] > 0]).astype(np.float64)
        tau = 0.01
        for _ in range(50):  # Newton steps for the 1-parameter logistic regression
            p = 1.0 / (1.0 + np.exp(-np.clip(tau * x, -50, 50)))
            grad = ((y - p) * x).sum()
            hess = (p * (1 - p) * x * x).sum() + 1e-12
            step = grad / hess
            tau = max(tau + step, 1e-6)
            if abs(step) < 1e-9 * max(tau, 1e-6):
                break
        return float(min(tau, 1.0))

    def merge_targets(self, lr, v):
        """Clone each clone merges into: aneuploid clones without differences merge into the earlier one."""
        k = len(lr)
        target = np.arange(k)
        for i in range(2, k):
            for j in range(1, i):
                if target[j] == j:
                    d = lr[i] - lr[j]
                    pmd, pnd, _, _ = self.hmm(d - np.median(d), v[i] + v[j])
                    if _burden(pmd[0], pnd[0], self.data.lay) < self.min_burden:
                        target[i] = j
                        break
        return target

    def run(self, assign, n_iter):
        n = len(assign)
        assign = _compact(assign)
        for it in range(n_iter):
            k = int(assign.max()) + 1
            lr, v = self.profiles(assign, k)
            target = self.merge_targets(lr, v)
            if (target != np.arange(k)).any():
                log.debug("copy_number: iteration %d, clones -> %s", it, target)
                assign = _compact(target[assign])
                continue
            mu, pn, st = self.fit(lr, v, cross=True)
            priors = self.type_priors(self.fit(lr, v, cross=False)[0])
            ll = self.loglik(mu, assign)
            # drop the aneuploid clone whose likelihood gain does not pay for its segments (BIC)
            gain = np.full(k, np.inf)
            n_seg = 1 + (np.diff(st, axis=1) != 0).sum(axis=1)
            for c in range(1, k):
                mine = assign == c
                rest = np.delete(ll[mine], c, axis=1).max(axis=1)
                gain[c] = (ll[mine, c] - rest).sum() - n_seg[c] * np.log(n)
            worst = int(np.argmin(gain))
            if gain[worst] < 0:
                ll[:, worst] = -np.inf
                log.debug("copy_number: iteration %d dropped clone %d (margin %.1f)", it, worst, gain[worst])
                assign = _compact(np.argmax(ll, axis=1))
                continue
            ll += np.log(priors)[self.types]
            new = np.argmax(ll, axis=1)
            small = np.bincount(new, minlength=k) < _MIN_CLONE
            small[0] = False
            if small.any():
                ll[:, small] = -np.inf
                new = np.argmax(ll, axis=1)
            changed = float((new != assign).mean())
            log.debug("copy_number: iteration %d, %d clones, %.4f of cells changed", it, k, changed)
            assign = _compact(new)
            if changed < 2e-3:
                break
        k = int(assign.max()) + 1
        lr, v = self.profiles(assign, k)
        mu, pn, st = self.fit(lr, v, cross=True)
        self.priors = self.type_priors(self.fit(lr, v, cross=False)[0])
        ll = self.loglik(mu, assign, report=True)
        llr = ll[:, 1:].max(axis=1) - ll[:, 0] if k > 1 else np.zeros(n)
        model, spill = self.model, self.spill
        self.tau = self.temperature(mu, assign) if self.calibrate else 1.0
        self.model, self.spill = model, spill
        log.debug("copy_number: likelihood temperature %.4g", self.tau)
        ll = self.tau * ll + np.log(self.priors)[self.types]
        resp = np.exp(ll - ll.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)
        mu, pn, st = self.fit(lr, v, cross=False)
        self.final = (mu, pn, st, lr, v)
        self.dosage = _dosage_weights(mu, lr, v, pn)
        return assign, resp, mu, llr


def _dosage_weights(mu, lr, v, pn, strength: float = _DOSAGE_PRIOR) -> np.ndarray:
    """Dosage sensitivity of each gene from the clone contrasts.

    Where an aneuploid clone's HMM profile is non-neutral, a gene's log ratio is regressed on the profile through the
    origin (precision-weighted, pooled over clones). The slope is shrunk towards 1 with the weight of `strength`
    segments' worth of evidence: genes that follow copy number fully get about 1, dosage-compensated genes less.
    """
    if len(mu) < 2:
        return np.ones(lr.shape[1])
    m, y, w = mu[1:], np.clip(lr[1:], -2, 2), 1.0 / v[1:]
    w = w * (pn[1:] < 0.5)
    num = (w * m * y).sum(axis=0)
    den = (w * m * m).sum(axis=0)
    prior = strength * np.median(den[den > 0]) if (den > 0).any() else 1.0
    return np.clip((num + prior) / (den + prior), 0.0, 2.0)


def _compact(assign):
    """Relabel to 0..k-1 without empty clones, keeping clone 0 (diploid) as 0 even when it is empty."""
    used = np.unique(assign)
    used = np.concatenate([[0], used[used != 0]])
    lut = np.full(int(assign.max()) + 1, -1)
    lut[used] = np.arange(len(used))
    return lut[assign]


def _block_size(lay) -> int:
    """Genes per block of the genomic statistics.

    `_BLOCK`, unless a typical chromosome has fewer than two blocks of `_BLOCK` genes (small panels): then an eighth
    of its genes (at least 4), so that blocks along a chromosome can still be compared.
    """
    g = int(np.median(lay.gs))
    return _BLOCK if g >= 2 * _BLOCK else max(4, g // 8)


def _block_medians(lr: np.ndarray, lay, size: int | None = None):
    size = _block_size(lay) if size is None else size
    vals, ch = [], []
    for c, (g0, g) in enumerate(zip(lay.g0s, lay.gs, strict=True)):
        nb = max(1, g // size)
        for part in np.array_split(np.arange(g0, g0 + g), nb):
            vals.append(np.median(lr[part]))
            ch.append(c)
    return np.array(vals), np.array(ch)


def _mode(lr: np.ndarray, lay) -> float:
    """Mode of the block medians of a profile (Gaussian kernel, bandwidth 0.05): the copy-neutral level.

    which the library size normalisation moves away from 0 when much of the genome is gained or lost.
    """
    x, _ = _block_medians(lr, lay)
    grid = np.linspace(np.quantile(x, 0.02), np.quantile(x, 0.98), 401)
    dens = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / 0.05) ** 2).sum(axis=1)
    return float(grid[np.argmax(dens)])


def _smooth_amplitude(lr: np.ndarray, lay, size: int | None = None) -> float:
    """Square root of the lag-1 autocovariance along the genome of the medians of blocks of `size` genes of a relative profile: the amplitude of its genomically smooth part.

    Square root of the lag-1 autocovariance along the genome of the medians of blocks of `size` genes of a relative
    profile: the amplitude of its genomically smooth part. Gene-level noise and expression differences between cell
    types are mostly independent between blocks and do not contribute; copy number changes span many blocks.
    """
    x, ch = _block_medians(lr, lay, size)
    x = x - np.median(x)
    same = np.diff(ch) == 0
    if not same.any():
        return 0.0
    return float(np.sqrt(max((x[1:] * x[:-1])[same].mean(), 0.0)))


# ---- gene layout and helpers (genomic order, bins, k-means) --------------------------------------------------


def _excluded_genes(var_names: pd.Index, exclude_genes: Sequence[str] | None) -> np.ndarray:
    if exclude_genes is None:
        names = var_names.astype(str)
        return names.str.startswith("HLA-") | names.isin(CELL_CYCLE_GENES)
    return var_names.isin(list(exclude_genes))


class _Layout:
    """Genes in genomic order and chromosomes, and the genomic bins of the outputs."""

    def __init__(self, var: pd.DataFrame, used: np.ndarray):
        sub = var.iloc[np.flatnonzero(used)]
        chromosomes = _natural_sort(list(sub["chromosome"].unique()))
        order, g0s, gs = [], [], []
        n = 0
        for c in chromosomes:
            idx = np.flatnonzero(used & (var["chromosome"] == c).to_numpy())
            idx = idx[np.argsort(var["start"].to_numpy()[idx], kind="stable")]
            order.append(idx)
            g0s.append(n)
            gs.append(len(idx))
            n += len(idx)
        self.chromosomes = chromosomes
        self.order = np.concatenate(order)
        self.col_map = np.full(len(var), -1, dtype=np.int64)
        self.col_map[self.order] = np.arange(n)
        self.g0s = np.array(g0s, dtype=np.int64)
        self.gs = np.array(gs, dtype=np.int64)
        self.n_genes = n
        self.chrom = var["chromosome"].to_numpy()[self.order].astype(str)
        self.start = var["start"].to_numpy()[self.order].astype(np.int64)
        self.end = var["end"].to_numpy()[self.order].astype(np.int64)

    def bins(self, seg: np.ndarray, bin_size: int) -> tuple[pd.DataFrame, dict[str, int], sp.csr_matrix]:
        """Genomic bins, the first bin of each chromosome, and the (bins x segments) matrix that averages the gene values of each bin.

        A bin without genes takes the value of the segment of the gene whose centre is nearest to the bin centre.
        """
        centre = (self.start + self.end) // 2
        rows, cols, vals, frames = [], [], [], []
        chr_pos: dict[str, int] = {}
        b0 = 0
        for c, g0, g in zip(self.chromosomes, self.g0s, self.gs, strict=True):
            cen, sg = centre[g0 : g0 + g], seg[g0 : g0 + g]
            length = max(HG38_LENGTHS.get(c, 0), int(self.end[g0 + g - 1]))
            nb = -(-length // bin_size)
            chr_pos[c] = b0
            starts = np.arange(nb, dtype=np.int64) * bin_size
            frames.append(
                pd.DataFrame({"chromosome": c, "start": starts, "end": np.minimum(starts + bin_size, length)})
            )
            gb = np.minimum(cen // bin_size, nb - 1)
            per_bin = np.bincount(gb, minlength=nb)
            rows.append(b0 + gb)
            cols.append(sg)
            vals.append(1.0 / per_bin[gb])
            empty = np.flatnonzero(per_bin == 0)
            if len(empty):
                mid = starts[empty] + bin_size // 2
                order = np.argsort(cen, kind="stable")
                pos = np.clip(np.searchsorted(cen[order], mid), 1, g - 1) if g > 1 else np.zeros(len(empty), int)
                left, right = order[np.maximum(pos - 1, 0)], order[pos]
                nearest = np.where(np.abs(cen[left] - mid) <= np.abs(cen[right] - mid), left, right)
                rows.append(b0 + empty)
                cols.append(sg[nearest])
                vals.append(np.ones(len(empty)))
            b0 += nb
        M = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(b0, int(seg.max()) + 1)
        )
        bins = pd.concat(frames, ignore_index=True)
        bins.index = bins.index.astype(str)
        return bins, chr_pos, M


def _csr(X) -> sp.csr_matrix:
    return X if sp.issparse(X) else sp.csr_matrix(X)


def _map(source, fn, rows=None):
    """Yield `(None, fn(X, row0, n_rows, None))` per block holding any of `rows` (sorted; `None`: all).

    The block's rows are `X[row0 : row0 + n_rows]`: the matrix itself if they are contiguous, else a copy of them.
    """
    for X, rws in source.rows_of(rows):
        X = _csr(X)
        if rws[-1] - rws[0] + 1 == len(rws):
            yield None, fn(X, int(rws[0]), len(rws), None)
        else:
            yield None, fn(X[rws], 0, len(rws), None)


def _subsample(idx: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    return idx if len(idx) <= n_max else np.sort(rng.choice(idx, n_max, replace=False))


def _kmeans(X: np.ndarray, k: int, rng) -> np.ndarray:
    """Cluster of each row of `X` after Lloyd's k-means fitted on at most `20 k` rows; no empty clusters."""
    sub = X[_subsample(np.arange(len(X)), 20 * k, rng)]
    centres = sub[rng.choice(len(sub), k, replace=False)]
    for _ in range(_KMEANS_ITER):
        a = _nearest(sub, centres)
        counts = np.bincount(a, minlength=k)
        new = np.zeros_like(centres)
        np.add.at(new, a, sub)
        filled = counts > 0
        centres[filled] = new[filled] / counts[filled, None]
    assign = _nearest(X, centres)
    return np.unique(assign, return_inverse=True)[1]


def _nearest(X: np.ndarray, centres: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    c2 = (centres**2).sum(axis=1)
    return np.concatenate(
        [np.argmin(c2[None, :] - 2 * X[i : i + chunk] @ centres.T, axis=1) for i in range(0, len(X), chunk)]
    )
