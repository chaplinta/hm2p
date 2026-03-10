clear all;

dirs = loadPCDirs();

animalsTable = readtable(dirs.metaAnimalsPath);
cellsTable = readtable(dirs.metaCellsPath);

refData = join(cellsTable, animalsTable, 'Keys', {'animal_id'});

nCells = height(cellsTable);

plotDPI = 300;
sF = 20000;

cellTypes = [];
allSpikeRates = [];
allSubTraces = [];
allSubTracesOnly = [];
allVssDiffIV = [];
allVssDiffRheo = [];
allVssDiffPassive = [];
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
            
    % Somehow this ends up in a struct with the same name.
    ephysData = ephysData.ephysData;

    posIndexes = ephysData.IV.stimvec >= 0;
    % Stim period is 1 second.
    spikeRates = ephysData.IV.spikecount(posIndexes);

    VssDiffIV = getSS(ephysData.IV);
    VssDiffRheo = getSS(ephysData.Rheobase);
    VssDiffPassive = getSS(ephysData.Passive);

    % Find all stims with no spikes up until the first one.
    spks = ephysData.IV.spikecount;
    stims = ephysData.IV.stimvec;
    traces = ephysData.IV.traces;
    noSpikeIndexesIV = getNoSpikeIndexes(spks);
    noSpikeIndexesRheo = getNoSpikeIndexes(ephysData.Rheobase.spikecount);
    noSpikeIndexesPassive = ones([1, size(ephysData.Passive.traces, 2)]);

    % Plot the IV traces
    traceIVPlotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-001-IV-trace.png"));    
    plot_intrinsic(ephysData.IV, 1, traceIVPlotPath);

    % Plot all sub threshold traces.
    plotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-002-IV-subtraces.png"));   
    
    fig = figure('visible', 'off');
    set(gcf, 'color', 'w');
    noSpikeTraces = traces(:, noSpikeIndexesIV==1);
    if ~isempty(noSpikeTraces)
        plot(noSpikeTraces, 'k', 'linewidth', 1);
    end

    hold on;
    scale_x = 0.1 * sF;
    scale_y = 10;
    pos_x = 0;
    if ~isempty(noSpikeTraces)
        pos_y = min(min(noSpikeTraces));
    else
        pos_y = 0;
    end
    drawScaleBar(scale_x, scale_y, pos_x, pos_y)
    hold off;
    
    set(gca, 'box', 'off');
    axis off;
    exportgraphics(fig, plotPath, 'Resolution', plotDPI);
    close(fig); 

    % IV plot = spikes per stim
    meanIVPlotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-003-IV-spikes.png"));    
    fig = figure('visible', 'off');
    plot(ephysData.IV.stimvec(posIndexes), ...
         spikeRates, ...
         '*-', 'linewidth', 2);
    xlabel('Injected current (pA)');
    ylabel('Spike rate (spikes/s)');
    box off;
    set(gca, 'TickDir', 'out');
    exportgraphics(fig, meanIVPlotPath, 'Resolution', plotDPI);
    close(fig); 
    
    % Steady state, taken from Simon's code.
    
    plotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-004-IV-SS.png"));   
    plotSS(ephysData.IV.stimvec(noSpikeIndexesIV==1), ...
           VssDiffIV(:, noSpikeIndexesIV==1), ...
           plotPath, plotDPI);


    % Plot the rheo traces
    traceRheoPlotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-100-Rheo-trace.png"));
    plot_intrinsic(ephysData.Rheobase, 2, traceRheoPlotPath);

    plotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-101-Rheo-SS.png"));   
    plotSS(ephysData.Rheobase.stimvec(noSpikeIndexesRheo==1), ...
           VssDiffRheo(:, noSpikeIndexesRheo==1), ...
           plotPath, plotDPI);

    plotPath = fullfile(dirs.ephysPlotDir, strcat(cellName, "-201-RMB-SS.png"));   
    plotSS(ephysData.Passive.stimvec(noSpikeIndexesPassive==1), ...
           VssDiffPassive(:, noSpikeIndexesPassive==1), ...
           plotPath, plotDPI);

    if length(spikeRates) == 19
        % Make sure this has the correct number of stims, otherwise you
        % can't average.
        allSpikeRates = [allSpikeRates; spikeRates];
        isPenkPos = string(refData.cell_type(iCell)) == "penkpos";
        cellTypes = [cellTypes isPenkPos];
        
        % Lazy for loop
        for iStim=1:size(traces, 2)
            baselineIndex = int32(round(0.1 * sF)) - 1;
            traces(:, iStim) = traces(:, iStim) - mean(traces(1:baselineIndex, iStim));
        end
    
        allSubTracesOnly = [allSubTracesOnly, traces(:, noSpikeIndexesIV==1)];

        stims(noSpikeIndexesIV==0) = NaN;

        traces(:, noSpikeIndexesIV==0) = NaN;
        allSubTraces = cat(3, allSubTraces, traces);

        VssDiffIV(:, noSpikeIndexesIV==0) = NaN;
        allVssDiffIV = cat(3, allVssDiffIV, VssDiffIV);

    else
        disp('Warning this cell does not have the correct number of IV stims');
    end

   
    VssDiffRheo(:, noSpikeIndexesRheo==0) = NaN;
    allVssDiffRheo = cat(3, allVssDiffRheo, VssDiffRheo);

    VssDiffPassive(:, noSpikeIndexesPassive==0) = NaN;
    allVssDiffPassive = cat(3, allVssDiffPassive, VssDiffPassive);
        
    

    
end

% Plot all cells plus the average    
meanIVPlotPath = fullfile(dirs.ephysPopPlotDir, "001-IV-all.png");

penkposMean = mean(allSpikeRates(cellTypes==1, :), 1);
penknegMean = mean(allSpikeRates(cellTypes==0, :), 1);

penkposSEM = std(allSpikeRates(cellTypes==1, :)) / sqrt(sum(cellTypes==1));
penknegSEM = std(allSpikeRates(cellTypes==0, :))  / sqrt(sum(cellTypes==0));

fig = figure('visible','off');
set(gcf,'color','w');

plot(ephysData.IV.stimvec(posIndexes), ...
     allSpikeRates(cellTypes==1, :)', 'Color', [0, 0.4470, 0.741, 0.3], 'linewidth', 0.5);
hold on;
plot(ephysData.IV.stimvec(posIndexes), ...
     allSpikeRates(cellTypes==0, :)', 'Color', [0.8500, 0.3250, 0.0980 0.3], 'linewidth', 0.5);
set(gca, 'ColorOrderIndex', 1)
errorbar(ephysData.IV.stimvec(posIndexes), ...
         penkposMean, ...
         penkposSEM, ...
         'linewidth', 2);
errorbar(ephysData.IV.stimvec(posIndexes), ...
         penknegMean, ...
         penknegSEM, ...
         'linewidth', 2);
hold off;
xlabel('Injected current (pA)');
ylabel('Spike rate (spikes/s)');
box off;
set(gca, 'TickDir', 'out');
exportgraphics(fig, meanIVPlotPath, 'Resolution', plotDPI);
close(fig); 

% Plot normalised version
meanIVPlotPath = fullfile(dirs.ephysPopPlotDir, "002-IV-all-norm.png");

allSpikeRatesNorm = allSpikeRates ./ repmat(max(allSpikeRates, [], 2), 1, size(allSpikeRates, 2));

penkposMeanNorm = mean(allSpikeRatesNorm(cellTypes==1, :), 1);
penknegMeanNorm = mean(allSpikeRatesNorm(cellTypes==0, :), 1);

penkposSEMNorm = std(allSpikeRatesNorm(cellTypes==1, :)) / sqrt(sum(cellTypes==1));
penknegSEMNorm = std(allSpikeRatesNorm(cellTypes==0, :))  / sqrt(sum(cellTypes==0));

fig = figure('visible', 'off');
set(gcf, 'color', 'w');
plot(ephysData.IV.stimvec(posIndexes), ...
     allSpikeRatesNorm(cellTypes==1, :)', 'Color', [0, 0.4470, 0.741, 0.3], 'linewidth', 0.5);
hold on;
plot(ephysData.IV.stimvec(posIndexes), ...
     allSpikeRatesNorm(cellTypes==0, :)', 'Color', [0.8500, 0.3250, 0.0980, 0.3], 'linewidth', 0.5);
set(gca, 'ColorOrderIndex', 1)
errorbar(ephysData.IV.stimvec(posIndexes), ...
         penkposMeanNorm, ...
         penkposSEMNorm, ...
         'linewidth', 2);
errorbar(ephysData.IV.stimvec(posIndexes), ...
         penknegMeanNorm, ...
         penknegSEMNorm, ...
         'linewidth', 2);
hold off;
xlabel('Injected current (pA)');
ylabel('Spike rate (normalised)');
box off;
set(gca, 'TickDir', 'out');
exportgraphics(fig, meanIVPlotPath, 'Resolution', plotDPI);
close(fig); 

% Plot the mean sub threshold traces.
meanTracesPenkPos = nanmean(allSubTraces(:, :, cellTypes==1), 3);
meanTracesPenkNeg = nanmean(allSubTraces(:, :, cellTypes==0), 3);

yMin = min(min([meanTracesPenkPos meanTracesPenkNeg]));
yMax = max(max([meanTracesPenkPos meanTracesPenkNeg]));

plotPath = fullfile(dirs.ephysPopPlotDir, "003-IV-all-subthresh-penkpos.png");
fig = figure('visible', 'off');
set(gcf, 'color', 'w');
plot(meanTracesPenkPos, 'Color', [0, 0.4470, 0.741, 1], 'linewidth', 1);
hold on;
% Plot transparent so it's the same size as penk neg plot.
plot(meanTracesPenkNeg, 'Color', [0.8500, 0.3250, 0.0980, 0], 'linewidth', 1);
annotation('rectangle',[0 0 1 1],'Color','w');
hold off;
ylim([yMin, yMax]);
% Scale bar
scale_x = 0.1 * sF;
scale_y = 10;
pos_x = 0;
pos_y = yMin;
drawScaleBar(scale_x, scale_y, pos_x, pos_y);
set(gca, 'box', 'off');
axis off;
exportgraphics(fig, plotPath, 'Resolution', plotDPI);
close(fig); 

plotPath = fullfile(dirs.ephysPopPlotDir, "004-IV-all-subthresh-penkneg.png");
fig = figure('visible', 'off');
set(gcf, 'color', 'w');
% Plot transparent so it's the same size as penk pos plot.
pp = plot(meanTracesPenkPos, 'Color', [0, 0.4470, 0.741, 0.1], 'linewidth', 1);
hold on;
plot(meanTracesPenkNeg, 'Color', [0.8500, 0.3250, 0.0980, 1], 'linewidth', 1);
annotation('rectangle',[0 0 1 1],'Color','w');
hold off;
ylim([yMin, yMax]);
% Scale bar
scale_x = 0.1 * sF;
scale_y = 10;
pos_x = 0;
pos_y = yMin;
drawScaleBar(scale_x, scale_y, pos_x, pos_y);
set(gca, 'box', 'off');
axis off;
exportgraphics(fig, plotPath, 'Resolution', plotDPI);
close(fig); 

% Plot steady state

plotPath = fullfile(dirs.ephysPopPlotDir, "005-IV-all-ss.png");
plotSSCellType(cellTypes, allVssDiffIV, ephysData.IV.stimvec, plotPath, plotDPI);

plotPath = fullfile(dirs.ephysPopPlotDir, "100-Rheo-all-ss.png");
plotSSCellType(cellTypes, allVssDiffRheo, ephysData.Rheobase.stimvec, plotPath, plotDPI);

plotPath = fullfile(dirs.ephysPopPlotDir, "200-Passive-all-ss.png");
plotSSCellType(cellTypes, allVssDiffPassive, ephysData.Passive.stimvec, plotPath, plotDPI);








% % Plot all sub threshold traces.
% plotPath = fullfile(dirs.ephysPopPlotDir, "003-IV-all-subthreshtraces.png");
% 
% fig = figure('visible', 'off');
% set(gcf, 'color', 'w');
% plot(allSubTracesOnly, 'linewidth', 0.5, 'Color', [0, 0, 0, 0.3]);
% 
% % Scale bar
% scale_x = 0.1 * sF;
% scale_y = 10;
% pos_x = 0;
% pos_y = min(min(min(allSubTracesOnly)));
% drawScaleBar(scale_x, scale_y, pos_x, pos_y)
% 
% set(gca, 'box', 'off');
% axis off;
% exportgraphics(fig, plotPath, 'Resolution', plotDPI);
% close(fig); 