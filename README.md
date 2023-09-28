# mccallExt

The paper is **mccallExt.pdf**, located at the top of the directory.

The code that generates the results shared in the paper is contained `cde/`.
The code generates data and figures are saved to `out/`.
In `out/`, datasets are prefixed with `dat_` and figures are prefixed with `fig_`.
Programs are numered by the order in which they should be run to reproduce the en.

Here is a description of the code:
* `mccall.py`: Module used in other Python code
* `00-ex-mccall-uniform-welfare.py`: Confirms numerical and theoretical findings (not used by paper); generates log file in `log/`
* `01-get-sequence-reservation-wages.py`: Generates sequences of reservation wages for plotting
* `02-fig-03-plot-sequence-reservation-wages.R`: Plots sequences of reservation wages generated by `01-get-sequence-reservation-wages.py`
* `03-mccall-uniform-parameters.py`: Generates parameters used in simulations
* `04-mccall-uniform-pr-ext.py`: Computes welfare statistics for different beliefs about the probability of an extension
* `05-mccall-uniform-length-ext.py`: Computes welfare statistics for different beliefs about the lenth of extension
* `06-figs-04-07-plot-simulations.R`: Plots welfare statistics and generates figures 4 through 7 in the paper
* `07-get-google-trends.R`: Queries Google Trends for different phrases
* `08-fig-01-plot-google-searches.R`: Plots queries, generates figure 1 in the paper

The Python scripts need to be run in order.
`03-mccall-uniform-parameters.py` before `04-mccall-uniform-pr-ext.py`.
All rely on `mccall.py`.

The simulations take a long time to run.
If the aim is reproducing the figures,
then the R code that generates the figures can be run
without running any Python code.
R code that generates figures relies on .csv files shared in `out/`.
To be clear: 
These datasets can be entirely reproduced by running the code as described.

There is one exception.
The Google Trends query in `cde/07-get-google-trends.R` may not return the exact results each run.
Thus, the dataset generated by this code is date-stamped.
But running `cde/08-fig-01-plot-google-searches.R` will reproduce the figure included with the paper in the repository.

A log file is saved to `log/`.

To run Python code,
I generally opened a Jupyter QtConsole,
navigated to the `cde/` directory, and
issued commands like `>>> run 00-ex-mccall-uniform-welfare.py`.

My installation of Python followed the steps described at [QuantEcon](https://quantecon.org/).
I relied on the following specifications:

|                     |                |
|---------------------|----------------|
| conda version       | 23.1.0         |
| conda-build version | 3.23.3         |
| python version      | 3.9.16.final.0 |


This public repository is locked. 


