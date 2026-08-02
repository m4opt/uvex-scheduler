all: fov/chips.ds9 fov/bounding-rectangle.ds9 fov/inscribed-circle.ds9 visualizations/fov.pdf visualizations/fov.mp4 visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap.mp4 visualizations/survey-footprints.pdf visualizations/expected-visits.pdf visualizations/skyblocks.pdf tables/fields.ecsv visualizations/time-utilization.pdf

fov/chips.ds9 fov/bounding-rectangle.ds9 fov/inscribed-circle.ds9 visualizations/fov.pdf visualizations/fov.mp4 &: notebooks/fov.ipynb
	jupyter execute $<

visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap.mp4 &: notebooks/skygrid.ipynb fov/bounding-rectangle.ds9 fov/inscribed-circle.ds9
	jupyter execute $<

visualizations/survey-footprints.pdf &: notebooks/survey-footprints.ipynb survey-footprints/lmlz-deep.ds9 survey-footprints/lmlz-wide.ds9 survey-footprints/magellanic-clouds.ds9
	jupyter execute $<

visualizations/expected-visits.pdf visualizations/skyblocks.pdf tables/fields.ecsv &: notebooks/skyblocks.ipynb fov/inscribed-circle.ds9
	jupyter execute $<

tables/initial-survey.ecsv: notebooks/main.ipynb tables/fields.ecsv fov/inscribed-circle.ds9
	jupyter execute $<

visualizations/time-utilization.pdf \
visualizations/cadence-distribution-2-bins.pdf \
visualizations/cadence-distribution-20-bins.pdf \
visualizations/cadence-distribution-4-bins.pdf \
visualizations/cadence-distribution-40-bins.pdf \
visualizations/cadence-distribution-8-bins.pdf \
&: notebooks/report.ipynb tables/initial-survey.ecsv fov/inscribed-circle.ds9 fov/bounding-rectangle.ds9 fov/chips.ds9
	jupyter execute $<
