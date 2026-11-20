# infercnvpy: Infer copy number variation (CNV) from single-cell and spatial transcriptomics data

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![PyPI][badge-pypi]][link-pypi]

[badge-tests]: https://github.com/scverse/infercnvpy/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/infercnvpy/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/infercnvpy
[badge-pypi]: https://img.shields.io/pypi/v/infercnvpy?logo=PyPI
[link-pypi]: https://pypi.org/project/infercnvpy/

infercnvpy is a scalable Python library to infer copy number variation (CNV) from single-cell and spatial
transcriptomics data.
It reimplements the method of R [inferCNV][], works with the [scverse ecosystem][], and can read large datasets
block by block.
[`tl.copy_number`][link-copy-number] also infers copy number profiles and clones, and labels each cell as aneuploid or
diploid.

[infercnv]: https://github.com/broadinstitute/inferCNV/wiki
[scverse ecosystem]: https://scverse.org/packages/#ecosystem

![Heatmap of the CNV of each cell along the chromosomes](https://raw.githubusercontent.com/scverse/infercnvpy/main/img/infercnv_heatmap.png)

The results look similar, but not identical, to those of R inferCNV.
We are happy about feedback and contributions.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [installation instructions][link-installation],
- [quickstart][link-quickstart],
- [tutorials][link-tutorials],
- [API documentation][link-api].

## Installation

You need Python 3.13 or newer. There are several options to install infercnvpy:

1. Install the latest release of `infercnvpy` from [PyPI][link-pypi]:

```bash
pip install infercnvpy
```

2. Install the latest development version:

```bash
pip install git+https://github.com/scverse/infercnvpy.git@main
```

## Release notes

See the [release notes on GitHub][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

See the [citation page][link-citation] of the documentation.

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/infercnvpy/issues
[changelog]: https://github.com/scverse/infercnvpy/releases
[link-docs]: https://infercnvpy.readthedocs.io/en/stable/
[link-api]: https://infercnvpy.readthedocs.io/en/stable/api.html
[link-copy-number]: https://infercnvpy.readthedocs.io/en/stable/generated/infercnvpy.tl.copy_number.html
[link-installation]: https://infercnvpy.readthedocs.io/en/stable/installation.html
[link-quickstart]: https://infercnvpy.readthedocs.io/en/stable/quickstart.html
[link-tutorials]: https://infercnvpy.readthedocs.io/en/stable/tutorials.html
[link-citation]: https://infercnvpy.readthedocs.io/en/stable/citation.html
