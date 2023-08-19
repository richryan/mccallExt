# mccallExt

The paper is **mccallExt.pdf**, located at the top of the directory.

The code that generates the results shared in the paper is contained `cde/`.
Data and figures are saved to `out/`.
In `out/`, datasets are prefixed with `dat_` and figures are prefixed with `fig_`.

Because the welfare simulations take a long time to run and the datasets are small,
the .csv files needed to reproduce the paper are committed to the repository.
These datasets can be entirely reproduced by running the code as described---with one exception.

A log file is saved to `log/`.

To replicate results,
code should -- for the most part -- be run in order of the program name.
The Python code needs to be run from the `cde/` directory because
it relies on relative paths.
I generally 
open a Jupyter QtConsole,
navigate to the `cde/` directory, and
issue commands like `>>> run 00-ex-mccall-uniform-welfare.py`.

My installation of Python followed the steps described at [QuantEcon](https://quantecon.org/).
I relied on the following specifications:

|                     |                |
|---------------------|----------------|
| conda version       | 23.1.0         |
| conda-build version | 3.23.3         |
| python version      | 3.9.16.final.0 |

Code is contained in the following files.

  * TKTK
