function dirs = loadPCDirs()

    baseCodePath = fileparts(mfilename('fullpath'));
    configIniFilePath = fullfile(baseCodePath, 'config.ini');
    if ~isfile(configIniFilePath)
        error("Please create a file called 'config.ini' in the root" + ...
            " directory of this repository with the following fields: " + ...
            "metadatadir, morphdir, analysisdir, ephysdir")
    end
    iniData = ini2struct(configIniFilePath);
    dirs.metaDataDir = iniData.metadatadir;
    dirs.morphDir = iniData.morphdir;
    dirs.ephysDir = iniData.ephysdir;
    dirs.processedDir = iniData.processeddir;
    dirs.analysisDir = iniData.analysisdir;
    
    dirs.metaAnimalsFileName = "animals.csv";
    dirs.metaCellsFileName = "cells.csv";
    dirs.tracingSubDir = 'Tracing';

    dirs.metaAnimalsPath = fullfile(dirs.metaDataDir, dirs.metaAnimalsFileName);
    dirs.metaCellsPath = fullfile(dirs.metaDataDir,dirs. metaCellsFileName);
    
    dirs.metricsCSVFile = fullfile(dirs.analysisDir, 'metrics.csv');
    dirs.statsSumCSVFile = fullfile(dirs.analysisDir, 'statsSum.csv');
    
    dirs.ephysPlotDir = fullfile(dirs.analysisDir, "ephys-plots");
    dirs.ephysPopPlotDir = fullfile(dirs.analysisDir, "ephys-plots-pop");
    dirs.morphPlotDir = fullfile(dirs.analysisDir, "morph-plots");
    dirs.morphPopPlotDir = fullfile(dirs.analysisDir, "morph-plots-pop");
    dirs.zprojsPlotDir = fullfile(dirs.analysisDir, "zprojs");
    dirs.zprojsMorphPlotDir = fullfile(dirs.analysisDir, "zprojs-morph-plot");
    dirs.statSumViolinDir = fullfile(dirs.analysisDir, "metric-violins");
    dirs.statSumHistDir = fullfile(dirs.analysisDir, "metric-hists");
    dirs.statSumBoxDir = fullfile(dirs.analysisDir, "metric-boxes");
    dirs.statSumBoxPtsDir = fullfile(dirs.analysisDir, "metric-boxes-pts");
    dirs.corrPlotDir = fullfile(dirs.analysisDir, "corr-plots");
    dirs.pcaPlotDir = fullfile(dirs.analysisDir, "pca-plots");
    
    [success, msg, msgid] = mkdir(dirs.ephysPlotDir);
    [success, msg, msgid] = mkdir(dirs.ephysPopPlotDir);
    [success, msg, msgid] = mkdir(dirs.analysisDir);
    [success, msg, msgid] = mkdir(dirs.morphPopPlotDir);
    [success, msg, msgid] = mkdir(dirs.morphPlotDir);
    [success, msg, msgid] = mkdir(dirs.statSumViolinDir);
    [success, msg, msgid] = mkdir(dirs.statSumHistDir);
    [success, msg, msgid] = mkdir(dirs.statSumBoxDir);
    [success, msg, msgid] = mkdir(dirs.statSumBoxPtsDir);    
    [success, msg, msgid] = mkdir(dirs.corrPlotDir);
    [success, msg, msgid] = mkdir(dirs.pcaPlotDir);