function plotSSCellType(cellTypes, ss, stimvec, plotPath, plotDPI)

    meanSSPenkPos = nanmean(ss(:, :, cellTypes==1), 3);
    meanSSPenkNeg = nanmean(ss(:, :, cellTypes==0), 3);
    
    semSSPenkPos = nanstd(ss(:, :, cellTypes==1), [], 3) / sqrt(sum(cellTypes==1));
    semSSPenkNeg = nanstd(ss(:, :, cellTypes==0), [], 3)  / sqrt(sum(cellTypes==0));
    
    
    fig = figure('visible', 'off');
    set(gcf, 'color', 'w');
    errorbar(stimvec, ...
             meanSSPenkPos, ...
             semSSPenkPos, ...
             'linewidth', 2);
    
    yLimVals = ylim();
    xline(0);
    yline(0);
    
    hold on;
    
    errorbar(stimvec, ...
             meanSSPenkNeg, ...
             semSSPenkNeg, ...
             'linewidth', 2);
    hold off;
    xlabel('Injected current (pA)');
    ylabel('Vm steady state (mV)');
    box off;
    set(gca, 'TickDir', 'out');
    exportgraphics(fig, plotPath, 'Resolution', plotDPI);
    close(fig); 

end