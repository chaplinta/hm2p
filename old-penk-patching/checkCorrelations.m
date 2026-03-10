% Sand box for testing for correlations.
xVar = "ephys_passive_rin"; %ephys_passive_rin % ephys_passive_rhreo %depth_pial
yVar = "ephys_passive_rhreo";

dirs = loadPCDirs();

metricsTable = readtable(dirs.metricsCSVFile);
%idxInclude = find(metricsTable.has_ephys & metricsTable.has_morph);
idxInclude = find(metricsTable.has_ephys);
metricsTable = metricsTable(idxInclude, :);

xData = metricsTable.(xVar);
yData = metricsTable.(yVar);

[r, p] = corr(xData, yData, 'Type', 'Spearman');

f = figure();
scatter(xData, yData, 'filled');
xlabel(xVar.replace('_', '\_'));
ylabel(yVar.replace('_', '\_'));
title(sprintf('r=%.2f p=%.3f', r, p))
