"""Per-session summaries for the standalone data-quality reports.

Each submodule turns one session's arrays into a small JSON-ready dict
(histograms, rates, downsampled traces). The S3 loop that feeds them is
``scripts/make_qc_reports.py``; the HTML pages are built by
``scripts/build_qc_reports_html.py``.

Submodules
----------
common     histogram, downsampling and encoding helpers
tracking   pose-tracker output (likelihoods, jumps, anatomical checks)
movement   kinematics.h5 (HD estimators, speed, AHV, occupancy)
syllables  keypoint-MoSeq syllable sequences
rois       ROI classifier outputs and features
spikes     dF/F baseline, event detection and CASCADE spike inference
"""
