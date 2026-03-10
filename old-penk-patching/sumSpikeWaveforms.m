
clear all;

dirs = loadPCDirs();

animalsTable = readtable(dirs.metaAnimalsPath);
cellsTable = readtable(dirs.metaCellsPath);

refData = join(cellsTable, animalsTable, 'Keys', {'animal_id'});

nCells = height(cellsTable);

plotDPI = 300;
sF = 20000;

allWaves = [];
cellTypes = [];
for iCell = 1:nCells
    disp(iCell)
    ephysId = string(refData.ephys_id(iCell));
    has_ephys = ~(ephysId == "");

    cellName = strcat(sprintf( '%03d', refData.cell_index(iCell)), "-", ...
                      string(refData.animal_id(iCell)), "-", ...
                      string(refData.slice_id(iCell)), "-", ...
                      string(refData.cell_slice_id(iCell)));

    cellNameDir = fullfile(dirs.processedDir, cellName);

    cellEPhysDataPath = fullfile(cellNameDir, 'ephys_data.mat');

    if ~(has_ephys && isfile(cellEPhysDataPath))
        continue;
    end

    ephysData = load(cellEPhysDataPath, 'ephysData');
    ephysData = ephysData.ephysData;
            
    % Off by a factor of 10 for some unknown reason.
    sIndex = int32(round(ephysData.active.params(18) * 10));
    sRKHz = 20;
    baseTimeMS = 1;
    preTimeMS = 7;
    postTimeMS = 20;
    spikePre = int32(round(preTimeMS * sRKHz));
    spikePost = int32(round(postTimeMS * sRKHz));
    spikeBase = int32(round(baseTimeMS * sRKHz));

    spikeWave = ephysData.active.trace(sIndex - spikePre:sIndex + spikePost);
    time = linspace(-preTimeMS, postTimeMS, length(spikeWave));

    % Plot the spike waveform.
    plotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-300-spkwave.png"));   
    
    fig = figure('visible', 'off');
    set(gcf, 'color', 'w');
    plot(time, spikeWave);
    xlim([-preTimeMS, postTimeMS]);


    hold on;
    scale_x = 1; %ms
    scale_y = 10; %mV
    pos_x = -preTimeMS;
    pos_y = mean(spikeWave(1:spikePre))/2;
    drawScaleBar(scale_x, scale_y, pos_x, pos_y)
    hold off;
    
    set(gca, 'box', 'off');
    axis off;
    exportgraphics(fig, plotPath, 'Resolution', plotDPI);
    close(fig); 

    allWaves = [allWaves; spikeWave'];
    isPenkPos = string(refData.cell_type(iCell)) == "penkpos";
    cellTypes = [cellTypes isPenkPos];

end


% Plot all spike waveforms.

penkposMean = mean(allWaves(cellTypes==1, :), 1);
penknegMean = mean(allWaves(cellTypes==0, :), 1);

penkposSEM = std(allWaves(cellTypes==1, :)) / sqrt(sum(cellTypes==1));
penknegSEM = std(allWaves(cellTypes==0, :))  / sqrt(sum(cellTypes==0));

plotPath = fullfile(dirs.ephysPopPlotDir, "300-spkwave-all-raw.png");   

fig = figure('visible', 'off');
set(gcf, 'color', 'w');
plot(time, allWaves(cellTypes==1, :), ...
    'Color', [0, 0.4470, 0.741, 0.3], 'linewidth', 0.5);
hold on;
plot(time, allWaves(cellTypes==0, :), ...
    'Color', [0.8500, 0.3250, 0.0980 0.3], 'linewidth', 0.5);

plot(time, penkposMean, ...
    'Color', [0, 0.4470, 0.741, 1], 'linewidth', 2);

plot(time, penknegMean, ...
    'Color', [0.8500, 0.3250, 0.0980 1], 'linewidth', 2);

hold off;

xlim([-preTimeMS, postTimeMS]);


hold on;
scale_x = 1; %ms
scale_y = 10; %mV
pos_x = -preTimeMS;
pos_y = mean(mean(allWaves(:, 1:spikeBase)))/2;
drawScaleBar(scale_x, scale_y, pos_x, pos_y)
hold off;

set(gca, 'box', 'off');
axis off;
exportgraphics(fig, plotPath, 'Resolution', plotDPI);
close(fig); 

% Plot all spike waveforms normalised.
allWavesMax = repmat(max(allWaves, [], 2), [1, size(allWaves, 2)]);
%allWavesMin = repmat(min(allWaves, [], 2), [1, size(allWaves, 2)]);
allWavesMin = repmat(mean(allWaves(:, 1:spikeBase), 2), [1, size(allWaves, 2)]);
allWavesNorm = (allWaves - allWavesMin) ./ (allWavesMax - allWavesMin);
penkposMean = mean(allWavesNorm(cellTypes==1, :), 1);
penknegMean = mean(allWavesNorm(cellTypes==0, :), 1);

penkposSEM = std(allWavesNorm(cellTypes==1, :)) / sqrt(sum(cellTypes==1));
penknegSEM = std(allWavesNorm(cellTypes==0, :))  / sqrt(sum(cellTypes==0));

plotPath = fullfile(dirs.ephysPopPlotDir, "302-spkwave-all-norm.png");   

fig = figure('visible', 'off');
set(gcf, 'color', 'w');
plot(time, allWavesNorm(cellTypes==1, :), ...
    'Color', [0, 0.4470, 0.741, 0.3], 'linewidth', 0.5);
hold on;
plot(time, allWavesNorm(cellTypes==0, :), ...
    'Color', [0.8500, 0.3250, 0.0980 0.3], 'linewidth', 0.5);

plot(time, penkposMean, ...
    'Color', [0, 0.4470, 0.741, 1], 'linewidth', 2);

plot(time, penknegMean, ...
    'Color', [0.8500, 0.3250, 0.0980 1], 'linewidth', 2);

hold off;

xlim([-preTimeMS, postTimeMS]);


hold on;
scale_x = 1; %ms
scale_y = 0; %mV
pos_x = -preTimeMS;
pos_y = 0.2;
drawScaleBar(scale_x, scale_y, pos_x, pos_y)
hold off;

set(gca, 'box', 'off');
axis off;
exportgraphics(fig, plotPath, 'Resolution', plotDPI);
close(fig); 

