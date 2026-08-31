SURVEY := 3_year_example_survey

all: $(SURVEY)/fov_plot.png \
     $(SURVEY)/fov.mp4 \
     $(SURVEY)/chips.ds9 \
     $(SURVEY)/bounding-rectangle.ds9 \
     $(SURVEY)/inscribed-circle.ds9 \
     $(SURVEY)/reduced-inscribed-circle.ds9 \
     $(SURVEY)/coverage_multiplicity_histograms.png \
     $(SURVEY)/fov_coverage_animation.gif \
     $(SURVEY)/skygrid_params.json \
     $(SURVEY)/survey-footprints.pdf \
     $(SURVEY)/expected_visits_map.pdf \
     $(SURVEY)/block_size_distribution.pdf \
     $(SURVEY)/block_partition_map.pdf \
     $(SURVEY)/fields.ecsv \
     $(SURVEY)/blocks.ecsv \
     $(SURVEY)/downlinks.ecsv \
     $(SURVEY)/block_availability.pdf \
     $(SURVEY)/initial-survey.ecsv \
     $(SURVEY)/initial-survey.mp4 \
     $(SURVEY)/time-utilization.pdf \
     $(SURVEY)/weekly-usage.pdf \
     $(SURVEY)/visited-fraction.ecsv \
     $(SURVEY)/visited-area.ecsv \
     $(SURVEY)/visit-multiplicity.pdf \
     $(SURVEY)/visit-map.pdf \
     $(SURVEY)/visited-fraction-cdf.ecsv \
     $(SURVEY)/visit-multiplicity-cdf.pdf \
     $(SURVEY)/survey-completeness.ecsv \
     $(SURVEY)/survey-completeness-over-time.pdf \
     $(SURVEY)/cadence-histogram.pdf \
     $(SURVEY)/pair-count-curve.pdf \
     $(SURVEY)/slew-duration-distribution.pdf \
     $(SURVEY)/slew-angle-distribution.pdf

$(SURVEY)/fov_plot.png \
$(SURVEY)/fov.mp4 \
$(SURVEY)/chips.ds9 \
$(SURVEY)/bounding-rectangle.ds9 \
$(SURVEY)/inscribed-circle.ds9 \
$(SURVEY)/reduced-inscribed-circle.ds9 &: notebooks/fov.ipynb
	jupyter execute $<

$(SURVEY)/coverage_multiplicity_histograms.png \
$(SURVEY)/fov_coverage_animation.gif \
$(SURVEY)/skygrid_params.json &: notebooks/skygrid.ipynb $(SURVEY)/chips.ds9 $(SURVEY)/inscribed-circle.ds9
	jupyter execute $<

$(SURVEY)/survey-footprints.pdf &: notebooks/survey-footprints.ipynb survey-footprints/lmlz-deep.ds9 survey-footprints/lmlz-wide.ds9 survey-footprints/magellanic-clouds.ds9
	jupyter execute $<

$(SURVEY)/expected_visits_map.pdf \
$(SURVEY)/block_size_distribution.pdf \
$(SURVEY)/block_partition_map.pdf \
$(SURVEY)/fields.ecsv \
$(SURVEY)/blocks.ecsv &: notebooks/skyblocks.ipynb notebooks/survey_utils.py $(SURVEY)/chips.ds9 $(SURVEY)/inscribed-circle.ds9 $(SURVEY)/skygrid_params.json
	jupyter execute $<

$(SURVEY)/downlinks.ecsv \
$(SURVEY)/block_availability.pdf \
$(SURVEY)/initial-survey.ecsv \
$(SURVEY)/initial-survey.mp4 &: notebooks/main.ipynb $(SURVEY)/chips.ds9 $(SURVEY)/inscribed-circle.ds9 $(SURVEY)/fields.ecsv $(SURVEY)/blocks.ecsv
	jupyter execute $<

$(SURVEY)/time-utilization.ecsv \
$(SURVEY)/time-utilization.pdf \
$(SURVEY)/weekly-usage.ecsv \
$(SURVEY)/weekly-usage.pdf \
$(SURVEY)/visited-fraction.ecsv \
$(SURVEY)/visited-area.ecsv \
$(SURVEY)/visit-multiplicity.pdf \
$(SURVEY)/visit-map.pdf \
$(SURVEY)/visited-fraction-cdf.ecsv \
$(SURVEY)/visit-multiplicity-cdf.pdf \
$(SURVEY)/survey-completeness.ecsv \
$(SURVEY)/survey-completeness-over-time.pdf \
$(SURVEY)/survey-completeness-over-time.ecsv \
$(SURVEY)/cadence-histogram.ecsv \
$(SURVEY)/cadence-histogram.pdf \
$(SURVEY)/pair-count-curve.ecsv \
$(SURVEY)/pair-count-curve.pdf \
$(SURVEY)/slew-duration-distribution.pdf \
$(SURVEY)/slew-duration-distribution.ecsv \
$(SURVEY)/slew-angle-distribution.pdf \
$(SURVEY)/slew-angle-distribution.ecsv &: notebooks/report.ipynb $(SURVEY)/initial-survey.ecsv $(SURVEY)/blocks.ecsv $(SURVEY)/bounding-rectangle.ds9 $(SURVEY)/inscribed-circle.ds9
	jupyter execute $<
