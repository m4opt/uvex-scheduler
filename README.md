# UVEX scheduler notebook and data products

- `main.ipynb`: Notebook to generate the data products below
- `fields.ecsv`: Working field grid and block definitions
- `initial-survey.ecsv`: Reference science timeline for first 100 days
- `fov.ds9`: Region file for camera footprint
- `fov-inscribed-circle.ds9`: Region file for circle inscribed within camera footprint
- `survey-footprints/*.ds9`: Regions that define survey footprints. Note that polygon edges are treated as great circle arcs, so if there are long straight edges in RA or Dec they must be subdivided into multiple edges.
