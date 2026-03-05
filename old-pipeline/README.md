# m2p

Install:

Install suite2p as described, e.g.

# If using apple m1/m2 chips, you need to run the terminal in rosetta mode, and use this command (see suite2p instructions)
```bash
CONDA_SUBDIR=osx-64 conda create --name suite2p python=3.8
```

# Otherwise
```bash
conda create --name suite2p python=3.8
```

Then
```bash
conda activate suite2p
python -m pip install suite2p'[all]'
```

(' quotes needed on OSX, not sure about others)


To test suite2p installation worked (seems to work with or without rosetta mode on OSX):

```bash
python -m suite2p
```

Next install ffmpeg binaries, e.g. on OSX: 

```bash
brew install ffmpeg
```

then either run this for exact versions

```bash
pip install -r requirements-exact.txt
```

or this for any (pipreqs not working on 3.8):

```bash
pip install numpy scipy pandas tables matplotlib imageio 'imageio[ffmpeg]' 'tifffile[all]' nptdms shapely pycircstat 
decorator nose napari ffmpeg-python jupyterlab plotly ipywidgets jupyter-dash seaborn statsmodels upsetplot umap-learn
```

Run pipfreeze to save exact versions;
```bash
pip freeze > requirements-exact.txt
```

To run scripts, from project root dir

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```


Data setup
1. Set paths in paths/config.py
2. Download raw data with scripts/copy_raw_local.py
3. Download video meta files and optional preprocessed video (see example-rsync.txt).

Video preprocessing
All video has been annotated, which defines a cropping region, a scale bar, and the region of interest (ROI, the 
floor of the maze). This can be modified with mov_annotate.py, one experiment at a time.
1. Either download pre processed video folder, or run
   1. mov_undistort.py
   2. mov_crop.py
3. Run copy_mov_dlc to copy all cropped files into a folder so they can be annoated with deeplabcut.
   4. Run DLC locally to annotate. Copy data to winstor to run on HPC.
   5. 

2p preprocessing
1. Run run_s2p.py to process using suite 2p.

Workflow for getting final data set:
1. Run scripts/raw2multi.py for all experiment ids that could be used. 2p data goes into inspect-2p folder or 
inspect-2p-check etc as needed. These can be deleted.
   1. Check that the data looks good. Note any bad frames (usually PMT dropout or MEMS failur) in 
metadata/experiments.csv.
2. Run scripts/mov_multi.py for all good 2p experiments. Movies go into inspect-behav folder.
   1. Check for periods when the mouse is stuck, note times in 
metadata/experiments.csv.
3. Run scripts/bad_periods.py to check that the bad and good periods of 2p and behaviour are correctly extracted.
4. Run check_experiments.py to check that all experiments have usable timings. This also finds experiments with 
   potentially bad 2p frames, analysis in saved in inspect-2p-plots.


Workflow for processing data:
1. Process raw data 
   1. Movies -> DLC (note requires preparation steps)
      1. mov_undistort
      2. mov_annotate
         1. Select crop region, line between holes on table (scale) and ROI at base of floor near walls, save all 
            layers as "meta".
      3. mov_crop: applies crop.
      4. mov_dlc: runs dlc.
   2. Images -> run_single_s2p.py
      1. Then run suite2p gui and manually check classification of soma and dendrites.
         2. Classification uses both morphphology and traces (F and neuropil) so you can exclude them if they have 
            bad traces, e.g. bad neuropil is bad, overfilled maybe. See Measures Used by Classifier https://github.
            com/cortex-lab/Suite2P/blob/master/README.md
      2. Then run s2p_class.py so as to generate list of training data for classification.
      3. Run suite2p gui (seems you also need to open a data set) then build  classifiers based on text files, save 
         as npy.
   
Raw data folders
Whenever accessed, the video folder is deleted meta data will be copied to the video_meta_bak_path, so that the video 
dir can be deleted and regenerated. In this scenario, meta data will be copied back from the backup directory to the 
video directory.

2. Process tracking and Ca data. Currently run.py:
   1. Calculates behavioural metrics.
   2. Resamples behaviour to imaging frames.
   3. Aggregates all in a single data frame.
   4. todo Calculate single cell statisitcs (HD etc), save tuning curves.
   5. If there are soma dend pairs, it analyses them.





Workflow for aggregation
1) Combine all processed datasets across animals and save. Pandas tables:
   1) Cell statistics
   2) Soma/dend statistics.
   3) Tuning curves: AHV, HD, place, speed.

Work flow for analysis:
1) Single cell:
   1) Plot tuning curves.
   2) Plot distributions of single cell statistics.
2) Soma/dend:
   1) Plot tuning curves.
   2) Plot distributions of single cell statistics.

