fov.ds9 fov-inscribed-circle.ds9 visualizations/fov.mp4 &: notebooks/fov.ipynb
	jupyter execute $<

visualizations/skygrid.mp4 visualizations/coverage-fraction.pdf visualizations/skygrid-overlap &: notebooks/skygrid.ipynb fov-inscribed-circle.ds9
	jupyter execute $<
