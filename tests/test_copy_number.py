"""`tl.copy_number`: HMM kernel against a dense reference, and the whole method on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from scipy.special import logsumexp

import infercnvpy as cnv
from infercnvpy.tl import _copy_number as M
from infercnvpy.tl import _copy_number_kernels as K

N_CHROM, GENES_PER_CHROM = 6, 400
#: (chromosome, first gene, last gene, copy number factor) of the tumour cells, for 400 genes per chromosome
EVENTS = [(0, 0, 200, 2.0), (2, 100, 400, 0.5), (4, 0, 400, 2.0)]


def _simulate(n_tumour=240, n_normal=160, depth=3000, seed=0, normal_depth=0.5, n_chrom=N_CHROM, genes=GENES_PER_CHROM):
    """Tumour cells with `EVENTS` (positions scaled to `genes` per chromosome) and two normal cell types."""
    rng = np.random.default_rng(seed)
    G = n_chrom * genes
    mu = rng.lognormal(-1, 1.2, G)
    cn = np.ones(G)
    for c, a, b, f in EVENTS:
        cn[c * genes + a * genes // GENES_PER_CHROM : c * genes + b * genes // GENES_PER_CHROM] = f
    n = n_tumour + n_normal
    tumour = np.r_[np.ones(n_tumour, bool), np.zeros(n_normal, bool)]
    # two normal cell types, with 5% of their genes 4x up or down
    programs = np.where(rng.random((3, G)) < 0.05, rng.choice([0.25, 4.0], (3, G)), 1.0)
    ctype = np.where(tumour, 0, 1 + rng.integers(0, 2, n))
    lam = mu[None, :] * programs[ctype] * np.where(tumour[:, None], cn[None, :], 1.0)
    # normal (immune, stromal) cells usually have fewer UMIs than tumour cells (`normal_depth`)
    lam *= (depth * np.where(tumour, 1.0, normal_depth) * rng.lognormal(0, 0.2, n) / lam.sum(axis=1))[:, None]
    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    pos = np.tile(np.arange(genes) * (80_000_000 // genes) + 10_000, n_chrom)
    var = pd.DataFrame(
        {"chromosome": np.repeat([f"chr{i + 1}" for i in range(n_chrom)], genes), "start": pos, "end": pos + 5_000},
        index=[f"g{i}" for i in range(G)],
    )
    obs = pd.DataFrame({"truth": np.where(tumour, "tumour", "normal")}, index=[f"c{i}" for i in range(n)])
    return AnnData(X, obs=obs, var=var)


def _forward_backward_dense(y, v, stay, means, log_prior, nu):
    """Textbook forward-backward with explicit transition matrices, in log space."""
    S, G = len(means), len(y)
    z = (y[:, None] - means[None, :]) ** 2 / (nu * v[:, None])
    emis = -0.5 * np.log(v)[:, None] - 0.5 * (nu + 1) * np.log1p(z)
    prior = np.exp(log_prior)
    la = np.empty((G, S))
    la[0] = log_prior + emis[0]
    for i in range(1, G):
        T = stay[i] * np.eye(S) + (1 - stay[i]) * prior[None, :]
        la[i] = logsumexp(la[i - 1][:, None] + np.log(T), axis=0) + emis[i]
    lb = np.zeros((G, S))
    for i in range(G - 2, -1, -1):
        T = stay[i + 1] * np.eye(S) + (1 - stay[i + 1]) * prior[None, :]
        lb[i] = logsumexp(np.log(T) + (emis[i + 1] + lb[i + 1])[None, :], axis=1)
    post = np.exp(la + lb - logsumexp(la[-1]))
    return post @ means, logsumexp(la[-1])


def test_hmm_kernel_matches_dense():
    rng = np.random.default_rng(0)
    G = 60
    means = np.log(np.array([0.5, 1.0, 1.5]))
    means[1] = 0.0
    log_prior = np.log(np.array([0.25, 0.5, 0.25]))
    y = np.r_[np.zeros(20), np.full(20, 0.4), np.zeros(20)] + rng.normal(0, 0.3, G)
    v = rng.uniform(0.05, 0.2, G)
    stay = np.full(G, 0.95)
    stay[0] = stay[30] = 0.0
    pm, pn, st, ll = (np.empty((1, G)), np.empty((1, G)), np.empty((1, G), dtype=np.int64), np.empty(1))
    K.hmm_posteriors(y[None], v[None], stay, means, log_prior, 2.0, pm, pn, st, ll)
    ref_pm, ref_ll = _forward_backward_dense(y, v, stay, means, log_prior, 2.0)
    np.testing.assert_allclose(pm[0], ref_pm, atol=1e-8)
    np.testing.assert_allclose(ll[0], ref_ll, rtol=1e-8)


@pytest.mark.parametrize(
    "kw",
    [{}, {"normal_depth": 1.0, "n_tumour": 600, "n_normal": 60}, {"n_tumour": 100, "n_normal": 500}],
)
def test_copy_number_synthetic(kw):
    adata = _simulate(**kw)
    cnv.tl.copy_number(adata, n_jobs=2)
    tumour = adata.obs["truth"].to_numpy() == "tumour"
    assert ((adata.obs["copy_number"] == "aneuploid").to_numpy() == tumour).mean() >= 0.98
    uns = adata.uns["copy_number"]
    X = adata.obsm["X_copy_number"]
    assert X.shape == (adata.n_obs, len(uns["bins"]))
    assert {"chromosome", "start", "end"} <= set(uns["bins"].columns)
    # the tumour pseudobulk follows the simulated events: gains positive, the loss negative
    bins = uns["bins"]
    prof = np.nanmean(X[tumour], axis=0)
    for c, a, b, f in EVENTS:
        chrom = f"chr{c + 1}"
        sel = (bins["chromosome"] == chrom).to_numpy() & (bins["start"] >= a * 200_000).to_numpy()
        sel &= (bins["end"] <= b * 200_000).to_numpy()
        assert np.sign(prof[sel].mean()) == np.sign(np.log(f))
    assert np.abs(np.nanmean(X[~tumour], axis=0)).max() < 0.05


def test_copy_number_small_panel_keeps_labels():
    """On a ~1,000-gene panel (45 genes per chromosome) the genomic statistics still have blocks to compare.

    With fixed 25-gene blocks, every chromosome had one block: flatness was 0 for every group and amplitudes were
    undefined, so the baseline seed fell on an arbitrary group (tumour and normal labels could swap) or every cell was
    called diploid (accuracy 0.4 here).
    """
    adata = _simulate(n_tumour=600, n_normal=400, depth=1000, n_chrom=22, genes=45)
    cnv.tl.copy_number(adata, n_jobs=2, min_genes=10, min_detection=0.01)
    tumour = adata.obs["truth"].to_numpy() == "tumour"
    assert ((adata.obs["copy_number"] == "aneuploid").to_numpy() == tumour).mean() >= 0.95


def test_copy_number_reference_and_not_inplace():
    adata = _simulate()
    obs, X, uns, gv = cnv.tl.copy_number(
        adata, reference_key="truth", reference_cat="normal", inplace=False, calculate_gene_values=True
    )
    assert gv.shape == adata.shape
    tumour_gv = np.nanmean(gv[(adata.obs["truth"] == "tumour").to_numpy()], axis=0)
    assert np.nanmean(tumour_gv[:200]) > 0.3  # chr1 0-200: gain (genes removed by the filters are NaN)
    assert uns["baseline"] == "reference"
    tumour = adata.obs["truth"].to_numpy() == "tumour"
    assert ((obs["copy_number"] == "aneuploid").to_numpy() == tumour).mean() >= 0.98
    assert "copy_number" not in adata.obs


@pytest.mark.parametrize("reference", [False, True])
def test_copy_number_fit_on_sample(reference):
    """With `max_fit_cells`, the model is fitted on a sample and all cells are assigned in one pass."""
    adata = _simulate(n_tumour=300, n_normal=200)
    kw = {"reference_key": "truth", "reference_cat": "normal"} if reference else {}
    kw["max_fit_cells"] = 250
    cnv.tl.copy_number(adata, n_jobs=2, **kw)
    tumour = adata.obs["truth"].to_numpy() == "tumour"
    assert ((adata.obs["copy_number"] == "aneuploid").to_numpy() == tumour).mean() >= 0.98
    assert adata.obs["copy_number_clone"].notna().all()


def test_smooth_amplitude_ignores_noise():
    adata = _simulate()
    from infercnvpy.tl._infercnv import _Rows

    excluded = M._excluded_genes(adata.var_names, None)
    _, lay = M._filter_cells_genes(adata, _Rows(adata.X, 5000), excluded, ("chrY",), 200, 0.05)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.5, lay.n_genes)
    step = np.where(np.arange(lay.n_genes) < lay.n_genes // 2, 0.3, -0.3)
    assert M._smooth_amplitude(noise, lay) < 0.05
    assert M._smooth_amplitude(noise + step, lay) > 0.2


def test_count_kernels_match_scipy():
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    X = sp.random(50, 30, density=0.3, format="csr", random_state=1, dtype=np.float32)
    X.data = np.round(X.data * 10)
    col_map = np.where(np.arange(30) % 7 == 3, -1, np.arange(30))  # genes 3, 10, 17, 24 dropped
    gene_win = np.repeat(np.arange(6), 5)
    out = np.empty((20, 6))
    K.window_counts(X.data, X.indices, X.indptr, np.arange(10, 30), col_map, gene_win, out, 3)
    Xk = X[10:30].toarray() * (col_map >= 0)
    win_of_col = np.where(col_map >= 0, gene_win[np.maximum(col_map, 0)], -1)
    np.testing.assert_allclose(out, np.stack([Xk[:, win_of_col == w].sum(1) for w in range(6)], axis=1))
    labels = rng.integers(0, 3, 20)
    slabs = np.zeros((4, 3, 30))
    K.pooled_counts(X.data, X.indices, X.indptr, np.arange(10, 30), col_map, labels, slabs)
    ref = np.zeros((3, 30))
    for lab in range(3):
        ref[lab, col_map[col_map >= 0]] = X[10:30][labels == lab].toarray()[:, col_map >= 0].sum(0)
    np.testing.assert_allclose(slabs.sum(0), ref)
    # the per-type likelihood equals the plain one per type
    Y = rng.poisson(5, (20, 6)).astype(float)
    size = Y.sum(1)
    share = rng.dirichlet(np.ones(6), (2, 3))
    phi = np.full(6, 3.0)
    types = rng.integers(0, 2, 20)
    ll = np.empty((20, 3))
    use = np.ones(6, dtype=np.bool_)
    g = rng.dirichlet(np.ones(6))
    spill = np.empty(20)
    K.nb_loglik_types(Y, size, types, share, phi, use, g, np.zeros(2, dtype=np.bool_), 0.5, False, ll, spill)
    for t in range(2):
        ref_ll = np.empty(((types == t).sum(), 3))
        K.nb_loglik(np.ascontiguousarray(Y[types == t]), size[types == t], share[t], phi, ref_ll)
        np.testing.assert_allclose(ll[types == t], ref_ll)
    # spillover: a cell whose counts mix its own shares with 30% of the aneuploid shares
    a = share[0, 0]
    y = 1e6 * (0.7 * a + 0.3 * g)
    np.testing.assert_allclose(K._spillover(y, a, g, use), 0.3, atol=1e-3)
    assert K._spillover(1e6 * a, a, g, use) < 1e-3
    # with spill_ok, the diploid likelihood uses the mixture (capped at 0.5), the aneuploid ones do not change
    ll2 = np.empty((20, 3))
    K.nb_loglik_types(Y, size, types, share, phi, use, g, np.ones(2, dtype=np.bool_), 0.5, False, ll2, spill)
    np.testing.assert_allclose(ll2[:, 1:], ll[:, 1:])
    s = np.minimum(spill, 0.5)
    for r in range(20):
        ref_ll = np.empty((1, 1))
        mix = ((1 - s[r]) * share[types[r], 0] + s[r] * g)[None]
        K.nb_loglik(Y[r : r + 1], size[r : r + 1], mix, phi, ref_ll)
        np.testing.assert_allclose(ll2[r, 0], ref_ll[0, 0])
