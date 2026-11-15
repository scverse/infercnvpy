from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from infercnvpy.pl._chromosome_heatmap import _finish

_METRICS = {
    "cells_gain": "cells to detect the gain",
    "cells_loss": "cells to detect the loss",
    "gain_ratio": "smallest detectable gain (copy ratio)",
    "loss_ratio": "mildest detectable loss (copy ratio)",
    "info": "information per cell",
    "null_var": "null variance",
}
_UNREACHABLE = "#bdbdbd"


def detectability(
    data: AnnData | pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    metric: Literal["cells_gain", "cells_loss", "gain_ratio", "loss_ratio", "info", "null_var"] = "cells_gain",
    key: str = "detectability",
    ideal: bool = False,
    kind: Literal["auto", "heatmap", "bar"] = "auto",
    min_genes: int = 5,
    cmap: str = "Blues",
    ax: plt.Axes | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool | None = None,
):
    """Plot the result of :func:`infercnvpy.tl.detectability` per chromosome arm.

    One condition (a group at a depth) gives a bar chart: a bar per arm for the achievable value, and a black tick for
    the ideal counting bound, so that the gap between them is visible. Several conditions (groups, depths, or several
    panels passed as a dict) give a heatmap with arms as columns and conditions as rows. Arms with fewer than
    `min_genes` panel genes are hatched, and values that no copy ratio or number of cells reaches are gray.

    Parameters
    ----------
    data
        AnnData with the result in `adata.uns[key]`, the table that :func:`infercnvpy.tl.detectability` returns, or a
        dict of such tables, e.g. one per panel (the keys label the rows).
    metric
        Column to plot: `cells_gain`, `cells_loss` (log scale), `gain_ratio`, `loss_ratio` (log scale),
        `info` or `null_var`.
    key
        Key of the result in `adata.uns`.
    ideal
        Plot the ideal counting bound (the `_ideal` columns) instead of the achievable value. Only for the heatmap;
        the bar chart shows both.
    kind
        `"bar"`, `"heatmap"`, or `"auto"`: a bar chart for one condition, a heatmap for several.
    min_genes
        Arms with fewer panel genes are hatched.
    cmap
        Color map of the heatmap.
    ax
        Axes to draw into.
    show
        Show the plot. Default: `scanpy.settings.autoshow`.
    save
        Save the figure as `<figdir>/detectability<save>`, as scanpy does.
    return_fig
        Return the figure.

    Returns
    -------
    The figure if `return_fig` is true, otherwise the axes if `show` is false.
    """
    if metric not in _METRICS:
        raise ValueError(f"`metric` must be one of {', '.join(_METRICS)}, got {metric!r}.")
    table = _table(data, key)
    rows = _conditions(table)
    if kind == "auto":
        kind = "bar" if rows.nunique() == 1 else "heatmap"
    if kind not in ("bar", "heatmap"):
        raise ValueError(f"`kind` must be 'auto', 'bar' or 'heatmap', got {kind!r}.")
    if kind == "bar" and rows.nunique() > 1:
        raise ValueError("A bar chart shows one condition; select one group and depth, or use `kind='heatmap'`.")
    arms = list(table["arm"].cat.categories if isinstance(table["arm"].dtype, pd.CategoricalDtype) else table["arm"])
    arms = list(dict.fromkeys(a for a in arms if a in set(table["arm"])))
    own = ax is None
    if ax is None:
        n_rows = 1 if kind == "bar" else rows.nunique()
        size = (max(6.0, 0.25 * len(arms) + 2), 3.5 if kind == "bar" else 0.35 * n_rows + 1.8)
        _, ax = plt.subplots(figsize=size, layout="constrained")
    if kind == "bar":
        _bars(ax, table, arms, metric, min_genes)
    else:
        column = f"{metric}_ideal" if ideal and metric not in ("info", "null_var") else metric
        _heatmap(ax, table, rows, arms, column, min_genes, cmap)
    fig = ax.get_figure(root=True)
    assert fig is not None
    return _finish(fig, ax, show=show, save=save, return_fig=return_fig, name="detectability", own=own)


def _table(data, key: str) -> pd.DataFrame:
    if isinstance(data, AnnData):
        if key not in data.uns or "table" not in data.uns[key]:
            raise ValueError(f"`adata.uns[{key!r}]` has no detectability table. Run `tl.detectability` first.")
        data = data.uns[key]["table"]
    if isinstance(data, Mapping):
        data = pd.concat([df.assign(panel=k) for k, df in data.items()])
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be an AnnData, a DataFrame or a dict of DataFrames.")
    return data.reset_index(drop=True)


def _conditions(table: pd.DataFrame) -> pd.Series:
    """A label per row from the columns that vary: panel, group and depth."""
    cols = [c for c in ("panel", "group", "depth") if c in table and table[c].nunique() > 1]
    if not cols:
        return pd.Series("", index=table.index)
    parts = [table[c].map(lambda d: f"{d:g} counts") if c == "depth" else table[c].astype(str) for c in cols]
    return pd.Series([" | ".join(p) for p in zip(*parts, strict=True)], index=table.index)


def _bars(ax: plt.Axes, table: pd.DataFrame, arms: list[str], metric: str, min_genes: int) -> None:
    log = metric.startswith("cells") or metric.endswith("ratio")
    t = table.set_index("arm").loc[arms]
    value = t[metric].to_numpy(dtype=float)
    ideal = t[f"{metric}_ideal"].to_numpy(dtype=float) if f"{metric}_ideal" in t else None
    ok = np.isfinite(value)
    x = np.arange(len(arms))
    base = 1.0 if log else 0.0
    bars = ax.bar(x[ok], value[ok] - base, bottom=base, width=0.7, color="#3a7dc9", label="achievable")
    for b, few in zip(bars, t["n_genes"].to_numpy()[ok] < min_genes, strict=True):
        if few:
            b.set_hatch("///")
            b.set_facecolor("#9ec2ea")
    if ideal is not None:
        good = np.isfinite(ideal)
        ax.scatter(x[good], ideal[good], marker="_", s=120, color="black", linewidths=2, label="ideal (counting only)")
    if log:
        ax.set_yscale("log")
    if (~ok).any():
        top = ax.get_ylim()[1]
        ax.scatter(x[~ok], np.full((~ok).sum(), top), marker="x", color=_UNREACHABLE, clip_on=False,
                   label="not reachable")  # fmt: skip
    ax.axhline(base, color="black", linewidth=0.8)
    ax.set_xticks(x, arms, rotation=90)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylabel(_METRICS[metric])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, _ = ax.get_legend_handles_labels()
    if (t["n_genes"].to_numpy() < min_genes).any():
        handles.append(Patch(facecolor="#9ec2ea", hatch="///", label=f"fewer than {min_genes} genes"))
    ax.legend(handles=handles, frameon=False, fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))


def _heatmap(ax, table, rows: pd.Series, arms: list[str], column: str, min_genes: int, cmap: str) -> None:
    labels = list(dict.fromkeys(rows))
    t = table.assign(_row=rows.to_numpy())
    grid = t.pivot_table(index="_row", columns="arm", values=column, aggfunc="first", observed=False, dropna=False)
    grid = grid.reindex(index=labels, columns=arms).to_numpy(dtype=float)
    few = t.pivot_table(index="_row", columns="arm", values="n_genes", aggfunc="first", observed=False, dropna=False)
    few = few.reindex(index=labels, columns=arms).to_numpy(dtype=float) < min_genes
    shown = np.abs(np.log2(grid)) if column.split("_ideal")[0].endswith("ratio") else grid
    finite = np.isfinite(shown)
    if column.startswith("cells") and finite.any():
        norm = mcolors.LogNorm(max(np.nanmin(shown[finite]), 1), max(np.nanmax(shown[finite]), 10))
    else:
        norm = None
    ax.set_facecolor(_UNREACHABLE)
    mesh = ax.pcolormesh(np.ma.masked_invalid(np.where(finite, shown, np.nan)), cmap=cmap, norm=norm,
                         edgecolors="white", linewidth=0.5)  # fmt: skip
    for i, j in zip(*np.nonzero(few), strict=True):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="///", edgecolor="#555555", linewidth=0))
    if grid.size <= 300:
        for i, j in zip(*np.nonzero(finite), strict=True):
            v = grid[i, j]
            text = f"{v:.2f}" if column.split("_ideal")[0].endswith("ratio") else f"{v:.3g}" if v < 1e4 else f"{v:.0e}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6, color=_ink(mesh, shown[i, j]))
    ax.set_xticks(np.arange(len(arms)) + 0.5, arms, rotation=90)
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    base = column.split("_ideal")[0]
    label = f"|log2|, {_METRICS[base].removesuffix(' (copy ratio)')}" if base.endswith("ratio") else _METRICS[base]
    label += " (ideal)" if column.endswith("_ideal") else ""
    fig = ax.get_figure(root=True)
    fig.colorbar(mesh, ax=ax, label=label, shrink=0.9)
    ax.legend(
        handles=[
            Patch(facecolor=_UNREACHABLE, label="not reachable"),
            Patch(facecolor="white", edgecolor="#555555", hatch="///", label=f"fewer than {min_genes} genes"),
        ],
        frameon=False,
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(0, -0.25 if len(arms) > 10 else -0.15),
        ncols=2,
    )


def _ink(mesh, value: float) -> str:
    """Black or white text, whichever reads on the cell's color."""
    r, g, b, _ = mesh.cmap(mesh.norm(value))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
