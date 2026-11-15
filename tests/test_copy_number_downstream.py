"""The outputs of `tl.copy_number` work with the downstream tools that read `tl.infercnv` outputs."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import scipy.sparse as sp

import infercnvpy as cnv
from tests.test_copy_number import _simulate


@pytest.fixture(scope="module")
def adata_cn():
    adata = _simulate()
    rng = np.random.default_rng(0)
    adata.obsm["spatial"] = rng.uniform(0, 100, (adata.n_obs, 2))
    # spatial neighbour graph: 6 nearest neighbours
    d = ((adata.obsm["spatial"][:, None] - adata.obsm["spatial"][None]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, 1:7]
    rows = np.repeat(np.arange(adata.n_obs), 6)
    adata.obsp["spatial_connectivities"] = sp.csr_matrix((np.ones(rows.size), (rows, nn.ravel())), (adata.n_obs,) * 2)
    cnv.tl.copy_number(adata, n_jobs=2)
    return adata


@pytest.mark.parametrize("mode", ["fast", "legacy"])
def test_chromosome_heatmaps(adata_cn, mode):
    cnv.pl.chromosome_heatmap(adata_cn, groupby="copy_number_clone", use_rep="copy_number", mode=mode, show=False)
    cnv.pl.chromosome_heatmap_summary(
        adata_cn, groupby="copy_number_clone", use_rep="copy_number", mode=mode, show=False
    )


def test_embedding_clustering_and_scores(adata_cn):
    a = adata_cn.copy()
    cnv.tl.pca(a, use_rep="copy_number", key_added="cn_pca")
    cnv.pp.neighbors(a, use_rep="cn_pca", key_added="cn_neighbors")
    cnv.tl.leiden(a, neighbors_key="cn_neighbors", key_added="cn_leiden")
    cnv.tl.umap(a, neighbors_key="cn_neighbors", key_added="cn_umap")
    assert a.obsm["X_cn_umap"].shape == (a.n_obs, 2)
    scores = cnv.tl.cnv_score(a, groupby="copy_number_clone", use_rep="copy_number", inplace=False)
    assert scores["T1"] > scores["N0"]
    ith = cnv.tl.ithcna(a, "truth", use_rep="X_copy_number", inplace=False)
    assert ith["tumour"] >= 0


def test_spatial(adata_cn):
    a = adata_cn.copy()
    cnv.tl.smooth_cnv(a, use_rep="copy_number", key_added="cn_smoothed")
    assert a.obsm["X_cn_smoothed"].shape == a.obsm["X_copy_number"].shape
    assert a.uns["cn_smoothed"]["chr_pos"] == a.uns["copy_number"]["chr_pos"]
    cnv.tl.spatial_regions(a, key="copy_number_clone", key_added="cn_regions")
    assert a.obs["cn_regions"].notna().all()
    cnv.pl.spatial_cnv(a, region="chr1:0-40000000", use_rep="copy_number", show=False)
