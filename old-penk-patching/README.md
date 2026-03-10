# penk-patching
Scripts for analysing data from penk patching experiments.

> Note: this repo was originally called patchcells so several files and 
> functions contain the acronym PC, this should be refactoed to PP for 
> penk-patching.

## Running the code:

The main file to run is procPC.m, which will process all data and summarise the data producing all plots and statistics. When run, it will ask if you want to reprocess the raw data, or load existing which saves quite a bit of time.

You can also run the individual summary scripts (prefix sum) to just do parts of the analyses.

Finally, there is also a small script that you can edit that plots correlations (checkCorrelations.m)

## Data:

The cleaned up raw data, including metadata and analyses, is stored at:

\\winstor.ad.swc.ucl.ac.uk\winstor\swc\margrie\tchapli\patching

The original raw confocal data (including tracings, full res images and properitery data formats) are stored at:

\\winstor.ad.swc.ucl.ac.uk\winstor\swc\margrie\Benoît\Research\Data\confocal-raw - 02022022

The original raw ephs data is stored at:

\\winstor.ad.swc.ucl.ac.uk\winstor\swc\margrie\SimonWeiler\RawData\Ephys_slice

## Requires:

morphology_tracing https://github.com/simonweiler/morphology_tracing

Violinplot https://github.com/bastibe/Violinplot-Matlab

ini2struct https://uk.mathworks.com/matlabcentral/fileexchange/17177-ini2struct

export_fig https://uk.mathworks.com/matlabcentral/fileexchange/23629-export_fig

WaveSurfer https://wavesurfer.janelia.org/



Statistics and Machine Learning Toolbox

Image Processing Toolbox

Signal Processing Toolbox

Curve Fitting Toolbox

### Setup:

Add all the directories in this repo to the path, and all the directories of the dependancies listed above.

Make an config.ini file in the root directory of this repository with the 
following fields: "metadatadir, morphdir, ephysdir, processeddir, analysisdir"

see example_config.ini in this repo.

where

metadatadir = The directory with the animals.csv. and cells.csv file (see 
bellow).

morphdir = The directory with cell morphologies. The top level folder is 
date of the imaging YYMMDD. The second folder is the date of the patching 
YYMMDD_S#_Hem, where S# is the slice number, e.g. S1, S2 etc, hem is the 
brain hemisphere (L, R or nothing [no underscore]). In the next directory, 
there is a folder called Tracing, and then another subdirectory for eac
CellN - e.g. Cell1, Cell2 etc. Each cell has a Apical.swc, Soma.swc and n
basal files - Basal1.swc, Basal2.swc etc. Finally, in the Tracing 
directory, the z-stack that was used for tracing should be saved as 
corrected.tif.

ephysdir = The directory with ephys raw. This follows Simon's naming convention, a folder for the date
YYMMDD then a folder for each cell, e.g. SW0001, then a h5 file for each protocol. ephys data is matched to morphology data with the animals and cells.csv file in the metadata directory.

processeddir = The directory where all processed data is stored.

analysisdir = The directory where all analysis output goes.

## Metadata
The metadata (animals and cells) is stored in the metadata directory in animals.csv and cells.csv

### animals.csv fields
- animal_id - mouse id in pyrate.
- sex - mouses' sex. 
- dob - mouse date of birth.
- date_slice - the date of slicing and ephys recording, important to match ephs to morphology.
- date_confocal - the date of confocal imaging, important to match ephs to morphology.

### cells.csv fields
- cell_index - unique numeric id of the cell in the project, so cells can be identified in plots etc.
- animal_id - the mouse id, must match to an animal_id in animals.csv
- slice_id - the slice number, as given when a slice is confocaled (e.g. first slice, second slice etc).
- cell_slice_id - a unique id of the cell within the slice.
- ephys_id - the name of the ephys recording of this cell
- cell_type - with it was a penk cell or not
- Trace-plot_id - not used, can be deleted
- hemisphere - the brain hemisphere the cell is in, e.g. left or right
- depth_slice - an estimated depth of the cell in the slice, from the top z-plane of the slice. Not actually needed because it can be obtained from the data procesing.
- depth_pial - an estimated depth of the cell from the cortical surface. Not actually needed because it can be obtained from the data procesing.
- depth_apical_tree - not used, can be deleted
- has_morph - whether there was any morphology data (i.e. slice not able to imaged)
- good_morph - whether the morphology data is usable (i.e. may be too hard to trace)
- area - the brain area, e.g. retrosplenial cortex
- layer - the cortical layer
- Coordinates (z ; y ; x)	- unsure, but appears to be estated location in the allen atlas. See Benoit's documents and thesis.
- quality	- comment on the quality of the tracing by Benoit
- matching - ? Made by Benoit.	
- Orientation	- orientation of the slice in the image. Not actually needed because it can be obtained from the data procesing.
- Axon - whether the axon is present. ot actually needed because it can be obtained from the data procesing.

## Outputs

### Cell metrics and population statistics

The metrics calculated for each cell (+ relevant meta data) are stored in analysis/metrics.csv. The statistical summary for all metrics (mean, std) are stored in sumStats.csv.

**todo** metric descriptions. There is currently some information in procPC.m, and in Benoit's thesis.

### Plots
Plots are stored in the relevant sub folder of analysis.

**todo** plot descriptions
