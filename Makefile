all: fov/chips.ds9 fov/bounding-rectangle.ds9 fov/inscribed-circle.ds9 visualizations/fov.pdf visualizations/fov.mp4 visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap.mp4 visualizations/survey-footprints.pdf visualizations/expected-visits.pdf visualizations/skyblocks.pdf tables/fields.ecsv

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
