clear all;

dirs = loadPCDirs();

disp('Calculating PCA and plotting ...')

metricsTable = readtable(dirs.metricsCSVFile);

penkposIndexes = startsWith(string(metricsTable.cell_type), 'penkpos');
penkposTable = metricsTable(penkposIndexes, :);

cellTypePrefixes = {'penkpos', 'penkneg'};

excludeDepthCols = {'morph_api_depth', 'morph_api_wd', ...
                    'morph_bas_depth', 'morph_bas_wd'};

% These values can depend on how much of the tree is visible in z, so
% exclude them.
exclude3DCols = [excludeDepthCols, ...
                {'morph_api_len', 'morph_bas_len', ... % total length
                 'morph_api_bpoints', 'morph_bas_bpoints', ... % number of branch points
                 'morph_api_shlpeakcr', 'morph_bas_shlpeakcr'}]; % sholl peak crossings


plotPCACellType(dirs, metricsTable, 0, 'Penkpos-Penkneg', cellTypePrefixes, excludeDepthCols);
plotPCACellType(dirs, penkposTable, 100, 'Penk-only', {}, excludeDepthCols);
 
plotPCACellType(dirs, metricsTable, 200, 'Penkpos-Penkneg-no3d', cellTypePrefixes, exclude3DCols);
plotPCACellType(dirs, penkposTable, 300, 'Penk-only-no3d', {}, exclude3DCols);

metricsTable = metricsTable(~ismember(metricsTable.cell_index, [19, 20, 33, 35]), :);

plotPCACellType(dirs, metricsTable, 400, 'Penkpos-Penkneg-noOutliers', cellTypePrefixes, exclude3DCols);
plotPCACellType(dirs, penkposTable, 500, 'Penk-only-noOutliers', {}, exclude3DCols);