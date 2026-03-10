function plotPCPCA(dataTable, tableIndexes, columPrefixes, plotTitle, plotDir, ...
                   cellTypeLabel, plotIndex, plotName, cellTypePrefixes, ...
                   excludeCols)

    if nargin < 8
        cellTypePrefixes = {};
    end
    if nargin < 9
        excludeCols = {};
    end

    plotDPI = 300;

    subDataTable = dataTable(tableIndexes, :);

    [matData, colNamesMat] = getMatrixFromTable(subDataTable, columPrefixes, excludeCols);    
    matData = normalize(matData);

    [coeffs, scores, latent, tsquared, explained, mu] = pca(matData);
    subplotIndex = 0;
    nPCsToPlot = 4;

    for iPC=1:2:nPCsToPlot
        subplotIndex = subplotIndex + 1;
        fig = figure('visible','off');
        colours = colororder();
        if isempty(cellTypePrefixes)
            scatter(scores(:, iPC), ...
                    scores(:, iPC+1), 'filled');
        else
            hold on;
            for iCellType=1:length(cellTypePrefixes)
                typeIndexes = startsWith(string(subDataTable.cell_type), cellTypePrefixes{iCellType}); 
                scatter(scores(typeIndexes, iPC), ...
                        scores(typeIndexes, iPC+1), ...
                        [], ...
                        colours(iCellType, :), ...
                        'filled');
            end
            hold off;
            legend(cellTypePrefixes);
        end
        %axis equal;
        xlabel(sprintf('%s principal component', iptnum2ordinal(iPC)));
        ylabel(sprintf('%s principal component', iptnum2ordinal(iPC+1)));
        set(gca, 'fontsize', 12);
        title(plotTitle);
        % displacement so the text does not overlay the data points
        dx = 0.01*range(scores(:,iPC)); dy = 0.01*range(scores(:,iPC+1)); 
        text(scores(:,iPC)+dx, ...
             scores(:,iPC+1)+dy, ...
             string(subDataTable.cell_index), ...
             'FontSize', 10);

        set(gca,'TickDir','out');
%         pbaspect([1 1 1])
    
        %saveas(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_%s.png', plotIndex, cellTypeLabel, subplotIndex, plotName)));
        exportgraphics(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_%s.png', plotIndex, cellTypeLabel, subplotIndex, plotName)), 'Resolution', plotDPI);
        close(fig); 

    end

    subplotIndex = subplotIndex + 1;
    fig = figure('visible','off');
    bar(explained);
    xlabel('PC Number');
    ylabel('Variance explained (%)');
    title(plotTitle);
    set(gca,'TickDir','out');
    set(gca,'box','off');
    %saveas(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_%s.png', plotIndex, cellTypeLabel, subplotIndex, plotName)));
    exportgraphics(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_%s.png', plotIndex, cellTypeLabel, subplotIndex, plotName)), 'Resolution', plotDPI);
    close(fig); 


    
    % Strip the ephs_ or morph_
    colNamesClean = {};
    for iCol=1:length(colNamesMat)
        colClean = replace(colNamesMat{iCol}, 'ephys_', '');
        colClean = replace(colClean, 'morph_', '');
        colNamesClean{end+1} = getPlainText(colClean);
    end
    for indexPC = 1: nPCsToPlot
        subplotIndex = subplotIndex + 1;
        fig = figure('visible','off');
        bar(coeffs(:, indexPC))
        xlabel('Feature');
        ylabel(sprintf('PC %d weight', indexPC));
        title(strcat(plotTitle, sprintf(' PC %d', indexPC)));
        set(gca,'xtick', [1:length(colNamesClean)]);
        set(gca, 'xticklabel', colNamesClean, 'fontsize', 6);
        xtickangle(90);
        set(gca,'TickDir','out');
        set(gca,'box','off');
        %saveas(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_PC-%02d_%s.png', plotIndex, cellTypeLabel, subplotIndex, indexPC, plotName)));
        exportgraphics(fig, fullfile(plotDir, sprintf('%03d_%s_%03d_PC-%02d_%s.png', plotIndex, cellTypeLabel, subplotIndex, indexPC, plotName)), 'Resolution', plotDPI);
        close(fig); 
    end


     


end
