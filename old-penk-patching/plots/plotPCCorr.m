function plotPCCorr(metricsTable, idxInclude, xVar, yVar, plotPath, iPlot)

    metricsTable = metricsTable(idxInclude, :);

    xData = metricsTable.(xVar);
    yData = metricsTable.(yVar);
    cellType = metricsTable.('cell_type');

    penkIndexes = ismember(cellType, 'penkpos');

    [r, p] = corr(xData, yData, 'Type', 'Spearman');

    f = figure('visible', 'off');
    scatter(xData(penkIndexes), yData(penkIndexes), 'filled');
    hold on;
    scatter(xData(~penkIndexes), yData(~penkIndexes), 'filled');
    hold off;

    % displacement so the text does not overlay the data points
    dx = 0.01*range(xData); dy = 0.01*range(yData); 
    text(xData+dx, yData+dy, string(metricsTable.cell_index), 'fontsize', 10);

    xlabel(xVar.replace('_', '\_'));
    ylabel(yVar.replace('_', '\_'));
    title(sprintf('r=%.2f p=%.3f', r, p));
    set(gca, 'fontsize', 12);

    %saveas(f, fullfile(plotPath, sprintf('%03d_%s_vs_%s.png', iPlot, xVar, yVar)));
    plotDPI = 300;
    exportgraphics(f, fullfile(plotPath, sprintf('%03d_%s_vs_%s.png', iPlot, xVar, yVar)), 'Resolution', plotDPI);   
    close(f); 

end