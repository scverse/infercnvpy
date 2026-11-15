"""numba kernels of :func:`infercnvpy.tl.detectability`, one pass over the CSR counts each.

A cell of share group `c` expects `m = s_a * p[c, g]` counts of gene `g` on arm `a`, with `s_a` the cell's leave-arm-out
size factor (`_sizes`). `_nb_moments` sums what the method-of-moments fit of the NB dispersion needs over the reference
cells, and `_arm_information` sums the Fisher information and the score moments per group and arm, plus the squared
efficient scores of the reference cells for the null variance.

The sums are per chunk of cells (`sums[t]`), so that the `prange` threads never write to the same memory, and the
scratch rows are allocated outside the `prange` loop, as in `_infercnv_kernels`.
"""

from __future__ import annotations

import numpy as np

from infercnvpy._numba import njit, njit_parallel, prange


@njit
def _sizes(data, indices, lo, hi, gene_arm, P, scale, out):
    """Leave-arm-out size factor of one cell (values `data[lo:hi]`) for every arm, into `out`.

    The counts off arm `a` divided by the share of those genes, `(N - O_a) / (1 - P[a])`, so that an arm's own counts
    (and a copy number change of it) don't enter its expectation. A panel that sits almost entirely on one arm falls
    back to the total `N`.
    """
    out[:] = 0.0
    total = 0.0
    for k in range(lo, hi):
        v = np.float64(data[k])
        total += v
        a = gene_arm[indices[k]]
        if a >= 0:
            out[a] += v
    for a in range(out.size):
        rest = 1.0 - P[a]
        out[a] = scale * ((total - out[a]) / rest if rest > 1e-3 else total)


@njit_parallel
def _nb_moments(data, indices, indptr, rows, code, P, p, gene_arm, xx, xm, s1, s2, buf):
    """Sums over the cells `rows` for the NB moments: per gene `x^2` and `x m`, per share group and arm `s` and `s^2`.

    `xx`, `xm` are (chunks, genes), `s1`, `s2` (chunks, share groups, arms) and `buf` (chunks, arms) is scratch.
    """
    n_chunks = xx.shape[0]
    n = rows.size
    for t in prange(n_chunks):
        sz = buf[t]
        for i in range(t * n // n_chunks, (t + 1) * n // n_chunks):
            r = rows[i]
            c = code[r]
            _sizes(data, indices, indptr[r], indptr[r + 1], gene_arm, P[c], 1.0, sz)
            for a in range(sz.size):
                s1[t, c, a] += sz[a]
                s2[t, c, a] += sz[a] * sz[a]
            for k in range(indptr[r], indptr[r + 1]):
                g = indices[k]
                a = gene_arm[g]
                if a >= 0:
                    v = np.float64(data[k])
                    xx[t, g] += v * v
                    xm[t, g] += v * sz[a] * p[c, g]


@njit_parallel
def _arm_information(
    data, indices, indptr, code, group, is_ref, P, p, phi, gene_arm, order, bounds, scale, sums, null, cell_info, buf
):
    """Per-cell Fisher information about the copy ratio of each arm, summed by group, and the null of the score.

    With `w = 1 / (1 + m phi)`, a cell's information on arm `a` is `I = sum m w`, and the variance of its efficient
    score `U = sum (x - m) w` at copy ratio `r` is `r A + r^2 B`, with `A = sum m w^2` and `B = sum m^2 phi w^2` (so
    `A + B = I`). `sums` (chunks, groups, 3, arms) gets `I`, `A`, `B`; `null` (chunks, 3, arms) gets `U^2`, `I` and
    `I^2` of the reference cells (`is_ref`); `cell_info` the information of each cell summed over arms. The sums run
    over the genes of each arm, `order[bounds[a]:bounds[a + 1]]`. `scale` multiplies the expected counts (a deeper
    experiment), which makes `U` meaningless: use `null` only from a call with `scale = 1`. `buf` (chunks, 2, arms) is
    scratch.
    """
    n_chunks = sums.shape[0]
    n_arms = bounds.size - 1
    n = indptr.size - 1
    for t in prange(n_chunks):
        sz = buf[t, 0]
        u = buf[t, 1]
        for r in range(t * n // n_chunks, (t + 1) * n // n_chunks):
            c = code[r]
            _sizes(data, indices, indptr[r], indptr[r + 1], gene_arm, P[c], scale, sz)
            u[:] = 0.0
            for k in range(indptr[r], indptr[r + 1]):
                g = indices[k]
                a = gene_arm[g]
                if a >= 0:
                    u[a] += np.float64(data[k]) / (1.0 + sz[a] * p[c, g] * phi[g])
            total = 0.0
            for a in range(n_arms):
                info = 0.0
                va = 0.0
                vb = 0.0
                for j in range(bounds[a], bounds[a + 1]):
                    g = order[j]
                    m = sz[a] * p[c, g]
                    w = 1.0 / (1.0 + m * phi[g])
                    info += m * w
                    va += m * w * w
                    vb += m * m * phi[g] * w * w
                sums[t, group[r], 0, a] += info
                sums[t, group[r], 1, a] += va
                sums[t, group[r], 2, a] += vb
                total += info
                if is_ref[r] and info > 0:
                    d = u[a] - info
                    null[t, 0, a] += d * d
                    null[t, 1, a] += info
                    null[t, 2, a] += info * info
            cell_info[r] = total
