# Numerical Examples for Structured $\mathcal{H}_2$-optimality Conditions

This repository contains code for numerical experiments reported in

> P. Mlinarić, P. Benner, S. Gugercin,
> **Interpolatory $\mathcal{H}_2$-optimality Conditions for Structured Linear
> Time-invariant Systems**,
> [*arXiv preprint*](https://arxiv.org/abs/2310.10618),
> 2023

## Installation

The code is implemented in the Python programming language
(tested using Python 3.10.12).

The necessary packages are listed in [`requirements.txt`](requirements.txt).
They can be installed in a virtual environment by, e.g.,

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Running the Experiments

The experiments are given as three Python scripts
(`second-order.py`, `port-Hamiltonian.py`, `time-delay.py`).
They can be opened as Jupyter notebooks via
[`jupytext`](https://jupytext.readthedocs.io/en/latest/).

## Author

Petar Mlinarić:

- affiliation: Virginia Tech
- email: mlinaric@vt.edu
- ORCiD: 0000-0002-9437-7698

## License

The code is published under the MIT license.
See [LICENSE](LICENSE).
