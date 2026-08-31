FOV_OUTPUTS = \
	fov/chips.ds9 \
	fov/bounding-rectangle.ds9 \
	fov/inscribed-circle.ds9 \
	visualizations/fov.pdf \
	visualizations/fov.mp4

SKYGRID_OUTPUTS = \
	visualizations/skygrid.mp4 \
	visualizations/coverage-fraction.pdf \
	visualizations/skygrid-overlap.mp4

SURVEY_FOOTPRINTS_OUTPUTS = \
	visualizations/survey-footprints.pdf

SKYBLOCKS_OUTPUTS = \
	visualizations/expected-visits.pdf \
	visualizations/skyblocks.pdf \
	tables/fields.ecsv
	tables/skyblocks.ecsv

MAIN_OUTPUTS = \
	tables/plan.ecsv

REPORT_OUTPUTS = \
	visualizations/time-utilization.pdf \
	visualizations/visit-distribution.pdf \
	visualizations/visit-map.pdf \
	visualizations/cadence-distribution-2-bins.pdf \
	visualizations/cadence-distribution-20-bins.pdf \
	visualizations/cadence-distribution-4-bins.pdf \
	visualizations/cadence-distribution-40-bins.pdf \
	visualizations/cadence-distribution-8-bins.pdf \
	visualizations/slew-angle-distribution.pdf

all: $(FOV_OUTPUTS) $(SKYGRID_OUTPUTS) $(SURVEY_FOOTPRINTS_OUTPUTS) $(SKYBLOCKS_OUTPUTS) $(MAIN_OUTPUTS) $(REPORT_OUTPUTS)

$(FOV_OUTPUTS) &: notebooks/fov.ipynb
	jupyter execute $<

$(SKYGRID_OUTPUTS) &: notebooks/skygrid.ipynb fov/bounding-rectangle.ds9 fov/inscribed-circle.ds9
	jupyter execute $<

$(SURVEY_FOOTPRINTS_OUTPUTS) &: notebooks/survey-footprints.ipynb survey-footprints/lmlz-deep.ds9 survey-footprints/lmlz-wide.ds9 survey-footprints/magellanic-clouds.ds9
	jupyter execute $<

$(SKYBLOCKS_OUTPUTS) &: notebooks/skyblocks.ipynb fov/inscribed-circle.ds9 notebooks/survey.py
	jupyter execute $<

$(MAIN_OUTPUTS): notebooks/main.ipynb tables/fields.ecsv fov/inscribed-circle.ds9 tables/skyblocks.ecsv
	jupyter execute $<

$(REPORT_OUTPUTS) &: notebooks/report.ipynb tables/plan.ecsv fov/inscribed-circle.ds9 fov/bounding-rectangle.ds9 fov/chips.ds9
	jupyter execute $<
