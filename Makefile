all: fov.ds9 fov-inscribed-circle.ds9 visualizations/fov.mp4 visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap.mp4 visualizations/survey-footprints.pdf

fov.ds9 fov-inscribed-circle.ds9 visualizations/fov.mp4 &: notebooks/fov.ipynb
	jupyter execute $<

visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap.mp4 &: notebooks/skygrid.ipynb fov-inscribed-circle.ds9
	jupyter execute $<

visualizations/survey-footprints.pdf &: notebooks/survey-footprints.ipynb survey-footprints/lmlz-deep.ds9 survey-footprints/lmlz-wide.ds9 survey-footprints/magellanic-clouds.ds9
	jupyter execute $<
