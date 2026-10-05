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

## To run

Run the following Jupyter notebooks, in order. To run a notebook interactively from your command line environment, just run `uv run jupyter execute path/to/notebook.ipynb`.

1.  [notebooks/fov.ipynb](notebooks/fov.ipynb): This saves the UVEX field of view geometry to region files in the [fov](fov) directory. It also generates some visualizations of the field of view.
2. [notebooks/skygrid.ipynb](notebooks/skygrid.ipynb): This generates some visualizations of the sky grid, including evaluating the sky coverage fraction.
3. [notebooks/survey-footprints.ipynb](notebooks/survey-footprints.ipynb): This generates a visualization of the survey footprint regions defined in the [survey-footprints](survey-footprints) directory.
4. [notebooks/skyblocks.ipynb](notebooks/skyblocks.ipynb): This groups the sky grid fields into blocks, assigns each block to a survey, and saves the field and block tables. It also generates visualizations of the expected number of visits and the sky blocks.
5. [notebooks/main.ipynb](notebooks/main.ipynb): This runs the scheduler to produce the reference science timeline.
6. [notebooks/report.ipynb](notebooks/report.ipynb): This generates summary visualizations of the schedule, including time utilization, cadence, sky coverage, and slew angle distributions.

Or, to run all of the notebooks automatically, you can run the following single command (note, require GNU Make):

    uv run make

## Contents

- `notebooks/*.ipynb`: Jupyter notebooks that generate the data files below.
- `notebooks/survey.py`: Survey configuration. Edit this to adjust survey footprints, required visits, cadence constraints, etc.
- `tables/fields.ecsv`: Working field grid and block definitions
- `tables/plan.ecsv`: Reference science timeline for a 2-year prime mission
- `fov/chips.ds9`: Region file for detector footprint accounting for chip gaps
- `fov/bounding-rectangle.ds9`: Region file for bounding rectangle enclosing all chips
- `fov/inscribed-circle.ds9`: Region file for circle inscribed within the bounding rectangle
- `survey-footprints/*.ds9`: Regions that define survey footprints. Note that polygon edges are treated as great circle arcs, so if there are long straight edges in RA or Dec they must be subdivided into multiple edges.

[uv]: https://docs.astral.sh/uv/
[M4OPT's instructions to install CPLEX]: https://m4opt.readthedocs.io/en/latest/install/cplex.html
