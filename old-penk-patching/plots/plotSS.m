function plotSS(stimvec, Vss, plotPath, plotDPI)
      
    fig = figure('visible', 'off');
    plot(stimvec, ...
         Vss, ...
         '*-', 'linewidth', 2);
    xlabel('Injected current (pA)');
    ylabel('Vm steady state (mV)');
    box off;
    set(gca, 'TickDir', 'out');
    exportgraphics(fig, plotPath, 'Resolution', plotDPI);
    close(fig); 
    
end