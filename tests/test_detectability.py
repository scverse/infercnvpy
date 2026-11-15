import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.stats import norm

import infercnvpy as cnv
from infercnvpy.tl._detectability import HG38_CENTROMERES

CHROMS = ["chr1", "chr2", "chr3", "chr4"]
CENTROMERES = {c: HG38_CENTROMERES[c] for c in CHROMS}
PHI = 0.5  # NB dispersion of every gene
ALPHA = 0.01
REPLICATES = 1000


class Sim:
    """NB counts of 15 genes on each arm of chr1 to chr4, with optional copy ratios and arm programs."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        rows = []
        for c in CHROMS:
            cen = HG38_CENTROMERES[c]
            for side, (lo, hi) in (("p", (1e6, cen - 1e6)), ("q", (cen + 1e6, cen + 40e6))):
                rows += [(c, int(s), c + side) for s in np.sort(self.rng.uniform(lo, hi, 15))]
        self.var = pd.DataFrame(rows, columns=["chromosome", "start", "arm"], index=[f"g{i}" for i in range(len(rows))])
        share = self.rng.lognormal(0, 1, len(rows))
        self.share = share / share.sum()

    def arm(self, name):
        return (self.var["arm"] == name).to_numpy()

    def counts(self, n, depth, *, ratio=None, program=None, fold=None):
        """`ratio` and `fold` ({arm: r}) multiply an arm's mean; `program` ({arm: sd}) by a random log-normal per cell."""
        m = depth * self.rng.lognormal(0, 0.3, n)[:, None] * self.share[None, :]
        for arm, sd in (program or {}).items():
            m[:, self.arm(arm)] *= np.exp(sd * self.rng.standard_normal(n) - sd**2 / 2)[:, None]
        for arm, r in {**(ratio or {}), **(fold or {})}.items():
            m[:, self.arm(arm)] *= r
        return self.rng.poisson(self.rng.gamma(1 / PHI, m * PHI))

    def adata(self, X, **obs):
        obs_df = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(X))])
        return AnnData(sp.csr_matrix(X.astype(np.float32)), obs=obs_df, var=self.var.drop(columns="arm"))

    def z_test(self, X, n, arm, v):
        """Score test of `arm` on clones of `n` cells (rows in order), with the true shares and the null variance `v`."""
        a = self.arm(arm)
        s = (X.sum(1) - X[:, a].sum(1)) / (1 - self.share[a].sum())  # leave-arm-out size, as the model's
        m = s[:, None] * self.share[a][None, :]
        w = 1 / (1 + m * PHI)
        U = ((X[:, a] - m) * w).sum(1).reshape(-1, n).sum(1)
        I = (m * w).sum(1).reshape(-1, n).sum(1)
        return U / np.sqrt(v * I)

    def power(self, depth, n, arm, r, v, **kwargs):
        z = self.z_test(self.counts(REPLICATES * n, depth, ratio={arm: r}, **kwargs), n, arm, v)
        return np.mean(z > norm.ppf(1 - ALPHA)) if r > 1 else np.mean(z < norm.ppf(ALPHA))


def _run(adata, **kwargs):
    kwargs = {"inplace": False, "alpha": ALPHA, "n_cells": 50, "centromeres": CENTROMERES} | kwargs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Measuring the null variance on all cells")
        return cnv.tl.detectability(adata, **kwargs).set_index("arm", drop=False)


@pytest.mark.parametrize("depth", [10, 30, 100])
def test_power_matches_simulation(depth):
    """A z-test on simulated clones reaches the predicted power, at the cells needed and at the smallest ratio."""
    sim = Sim()
    row = _run(sim.adata(sim.counts(3000, depth))).loc["chr1p"]
    v = max(row["null_var"], 1)
    assert 1 <= v < 1.4  # counting noise and the noise of the size factor only
    null = sim.z_test(sim.counts(REPLICATES * 50, depth), 50, "chr1p", v)
    assert np.mean(null > norm.ppf(1 - ALPHA)) < 2.5 * ALPHA
    for r, n in (
        (1.5, round(row["cells_gain"])),
        (0.5, round(row["cells_loss"])),
        (row["gain_ratio"], 50),
        (row["loss_ratio"], 50),
    ):
        assert sim.power(depth, n, "chr1p", r, v) == pytest.approx(0.8, abs=0.06), (r, n)


def test_null_variance_from_arm_program():
    """A program that moves an arm's genes together inflates its null variance; the ideal bound over-promises."""
    sim = Sim()
    program = {"chr1p": 0.3}
    df = _run(sim.adata(sim.counts(3000, 100, program=program)))
    row = df.loc["chr1p"]
    assert row["null_var"] > 1.5
    assert df.loc[df.arm != "chr1p", "null_var"].max() < 1.4
    assert row["cells_gain"] > 1.4 * row["cells_gain_ideal"]
    v = row["null_var"]
    achievable = sim.power(100, round(row["cells_gain"]), "chr1p", 1.5, v, program=program)
    ideal = sim.power(100, round(row["cells_gain_ideal"]), "chr1p", 1.5, v, program=program)
    assert achievable == pytest.approx(0.8, abs=0.06)
    assert ideal < 0.72


def test_cell_type_programs():
    """Without `groupby`, a type difference on an arm is null variance; with it, it is part of the expectation."""
    sim = Sim()
    X = np.vstack([sim.counts(1500, 50), sim.counts(1500, 50, fold={"chr2p": 4.0})])
    adata = sim.adata(X, cell_type=["A"] * 1500 + ["B"] * 1500)
    pooled = _run(adata)
    typed = _run(adata, groupby="cell_type")
    assert pooled.loc["chr2p", "null_var"] > 2
    assert (typed.loc["chr2p", "null_var"] < 1.4).all()
    assert set(typed["group"]) == {"A", "B"}
    b = typed[typed.group == "B"].loc["chr2p"]
    a = typed[typed.group == "A"].loc["chr2p"]
    assert b["info"] > 2 * a["info"]  # more counts on the arm in type B
    assert b["cells_gain"] < a["cells_gain"]


def test_reference_cells():
    """Copy number changes of non-reference cells don't enter the null variance."""
    sim = Sim()
    X = np.vstack([sim.counts(1500, 50), sim.counts(1500, 50, ratio={"chr1p": 2.0})])
    adata = sim.adata(X, status=["normal"] * 1500 + ["tumor"] * 1500)
    with pytest.warns(UserWarning, match="null variance on all cells"):
        everyone = cnv.tl.detectability(adata, inplace=False, centromeres=CENTROMERES).set_index("arm")
    ref = _run(adata, reference_key="status", reference_cat="normal")
    assert everyone.loc["chr1p", "null_var"] > 1.5
    assert ref.loc["chr1p", "null_var"] < 1.4


def test_inplace_and_obs():
    sim = Sim()
    adata = sim.adata(sim.counts(500, 30), status=["normal"] * 500)
    kwargs = {"reference_key": "status", "reference_cat": "normal", "centromeres": CENTROMERES}
    assert cnv.tl.detectability(adata, key_added="det", **kwargs) is None
    table = adata.uns["det"]["table"]
    assert list(table["arm"].astype(str)) == ["chr1p", "chr1q", "chr2p", "chr2q", "chr3p", "chr3q", "chr4p", "chr4q"]
    assert table["n_genes"].eq(15).all()
    assert adata.uns["det"]["params"]["n_reference"] == 500
    info = adata.obs["det_info"].to_numpy()
    assert info.shape == (500,)
    assert np.all(info >= 0)
    assert info.sum() == pytest.approx(500 * table["info"].sum())


def test_default_arms():
    """GRCh38 by default: all arms except the acrocentric p arms, with 0 genes where the panel has none."""
    sim = Sim()
    df = _run(sim.adata(sim.counts(300, 30)), centromeres=None)
    assert len(df) == 39
    assert "chr13p" not in df.index
    empty = df.loc["chr22q"]
    assert empty["n_genes"] == 0
    assert np.isinf(empty["cells_gain"])
    assert np.isnan(empty["gain_ratio"])


def test_gene_list_mode():
    sim = Sim()
    expression = sim.adata(sim.counts(1000, 50), status=["normal"] * 1000)
    genes = list(expression.var_names[::2])
    kwargs = {"reference_key": "status", "reference_cat": "normal"}
    with pytest.warns(UserWarning, match="1 of 61 panel genes"):
        from_list = _run(genes + ["NOT_A_GENE"], expression=expression, **kwargs)
    from_data = _run(expression[:, genes].copy(), **kwargs)
    pd.testing.assert_frame_equal(from_list, from_data)
    # the panel has fewer counts per cell than the transcriptome it is cut from
    assert from_list["counts_per_cell"].iloc[0] < np.median(expression.X.sum(axis=1).A1)


def test_depths():
    sim = Sim()
    adata = sim.adata(sim.counts(3000, 40))
    observed = np.median(adata.X.sum(axis=1).A1)
    df = _run(adata, counts_per_cell=[observed / 2, observed, observed * 2])
    by_depth = {d: g.set_index("arm") for d, g in df.groupby("depth")}
    low, mid, high = (by_depth[d] for d in sorted(by_depth))
    assert set(low["null_var_source"]) == set(mid["null_var_source"]) == {"measured"}
    assert set(high["null_var_source"]) == {"extrapolated"}
    ratio_low = low["cells_gain_ideal"] / mid["cells_gain_ideal"]
    ratio_high = mid["cells_gain_ideal"] / high["cells_gain_ideal"]
    assert ratio_low.between(1.3, 2.3).all()
    assert ratio_high.between(1.3, 2.3).all()
    assert (high["null_var"] >= mid["null_var"]).all()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_cells": 0}, ValueError, "n_cells"),
        ({"gain": 0.9}, ValueError, "gain"),
        ({"loss": 1.0}, ValueError, "loss"),
        ({"alpha": 0}, ValueError, "alpha"),
        ({"power": 1}, ValueError, "power"),
        ({"counts_per_cell": -5}, ValueError, "counts_per_cell"),
        ({"groupby": "nope"}, ValueError, "groupby"),
        ({"reference_key": "status"}, ValueError, "both"),
        ({"reference_key": "nope", "reference_cat": "x"}, ValueError, "reference_key"),
        ({"reference_key": "status", "reference_cat": "x"}, ValueError, "not found"),
        ({"layer": "counts"}, ValueError, "layer"),
        ({"expression": "x"}, ValueError, "expression"),
        ({"exclude_chromosomes": CHROMS}, ValueError, "No gene"),
    ],
)
def test_argument_errors(kwargs, error, match):
    sim = Sim()
    adata = sim.adata(sim.counts(50, 30), status=["normal"] * 50)
    with pytest.raises(error, match=match):
        _run(adata, **kwargs)


def test_input_errors():
    sim = Sim()
    adata = sim.adata(sim.counts(50, 30))
    with pytest.raises(ValueError, match="raw counts"):
        _run(AnnData(adata.X * 0.5, var=adata.var))
    with pytest.raises(ValueError, match="chromosome"):
        _run(AnnData(adata.X))
    with pytest.raises(ValueError, match="pass `expression`"):
        _run(["g0", "g1"])
    with pytest.raises(TypeError, match="gene symbols"):
        _run("g0")
    with pytest.raises(ValueError, match="None of the genes"):
        _run(["x"], expression=adata)


@pytest.fixture
def table():
    sim = Sim()
    X = np.vstack([sim.counts(300, 30), sim.counts(300, 30, fold={"chr2p": 3.0})])
    return _run(sim.adata(X, cell_type=["A"] * 300 + ["B"] * 300), groupby="cell_type", centromeres=None)


@pytest.mark.parametrize("metric", ["cells_gain", "loss_ratio", "null_var"])
def test_plot_heatmap(table, metric):
    ax = cnv.pl.detectability(table, metric=metric, show=False)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]
    assert len(ax.get_xticklabels()) == 39
    plt.close("all")


def test_plot_bar_and_panels(table):
    one = table[table.group == "A"]
    ax = cnv.pl.detectability(one, show=False)
    assert ax.get_yscale() == "log"
    assert len(ax.patches) == 8  # one bar per arm with genes
    ax = cnv.pl.detectability({"panel 1": one, "panel 2": one}, metric="gain_ratio", ideal=True, show=False)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["panel 1", "panel 2"]
    plt.close("all")


def test_plot_errors(table):
    with pytest.raises(ValueError, match="metric"):
        cnv.pl.detectability(table, metric="nope")
    with pytest.raises(ValueError, match="one condition"):
        cnv.pl.detectability(table, kind="bar")
    with pytest.raises(ValueError, match="Run `tl.detectability`"):
        cnv.pl.detectability(AnnData(np.zeros((2, 2))))
