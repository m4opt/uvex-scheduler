# UVEX scheduler notebook and data products

## Installation

1.  Install uv.

    We use [uv] to manage the installation of the dependencies. Install uv by
    running the command in the [official uv installation instructions]:

        curl -LsSf https://astral.sh/uv/install.sh | sh

    Then log out, and log back in.

2.  Clone this repository:

        git clone https://github.com/m4opt/uvex-scheduler.git
        cd uvex-scheduler

3.  Use uv to create a virtual environment and install all of the Python
    dependencies inside it by running the following command:

        uv sync

4.  Install CPLEX inside the project's uv virtual environment by following
    [M4OPT's instructions to install CPLEX].

## To rerun

After editing any of the Juptyer notebooks, run the following command to
regenerate any affected plots or data files (note, require GNU Make):

    uv run make

## Contents

- `notebooks/*.ipynb`: Jupyter notebooks that generate the data files below.
- `tables/fields.ecsv`: Working field grid and block definitions
- `initial-survey.ecsv`: Reference science timeline for first 100 days
- `fov/chips.ds9`: Region file for detector footprint accounting for chip gaps
- `fov/bounding-rectangle.ds9`: Region file for bounding rectangle enclosing all chips
- `fov/inscribed-circle.ds9`: Region file for circle inscribed within the bounding rectangle
- `survey-footprints/*.ds9`: Regions that define survey footprints. Note that polygon edges are treated as great circle arcs, so if there are long straight edges in RA or Dec they must be subdivided into multiple edges.

[uv]: https://docs.astral.sh/uv/
[M4OPT's instructions to install CPLEX]: https://m4opt.readthedocs.io/en/latest/install/cplex.html
