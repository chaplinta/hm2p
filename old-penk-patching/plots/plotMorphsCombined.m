function plotMorphsCombined(dirs, savePath, plotApical, plotBasal, plotSurface, plotAxon, cellTypePrefix, densityPlot)

    if nargin < 7
        cellTypePrefix = '';
    end

    if nargin < 8
        densityPlot = false;
    end

    if isfile(savePath) && ~densityPlot
        % Dendrite plots take forever.
        return;
    end

    fig=figure('visible','off');
    if ~densityPlot
        mon_pos=get(0,'MonitorPositions');
        set(gcf,'color','w', 'menubar','figure', 'position', [mon_pos(1,3)-1200 2 500 500]); % [left, bottom, width, height]
        hold on;
    end

    metricsTable = readtable(dirs.metricsCSVFile);

    if ~isempty(cellTypePrefix)
        cellTypeIndexes = startsWith(string(metricsTable.cell_type), cellTypePrefix); 
        metricsTable = metricsTable(cellTypeIndexes, :);
    end


    maxXAbs = 300;
    minYSurf = 200;
    maxYSurf = 200;
    binSize = 5;
    xEdges = [-maxXAbs:binSize:maxXAbs];
    yEdges = [-maxXAbs:binSize:maxXAbs];
    densitySigma = 1.5;
    
    allX = [];
    allY = [];
    densityMat = zeros(length(xEdges) - 1, length(xEdges) - 1);
    plotWidth = 1;
    nCells = 0;
    for iRow=1:height(metricsTable)

        if metricsTable.has_morph_good(iRow)

            cellX = [];
            cellY = [];

            cellName = string(metricsTable.proc_name(iRow));
            cellNameDir = fullfile(dirs.processedDir, cellName);
            cellMorphDataPath = fullfile(cellNameDir, 'morph_data.mat');

            disp(cellName);
        
            morphData = load(cellMorphDataPath, 'morphData');
            % Somehow this ends up in a struct with the same name.
            morphData = morphData.morphData;
        
            apical_tree = morphData.traces{1};
            com_tree = morphData.traces{2};
            soma_tree = morphData.traces{3};
            
            surface_tree = morphData.traces_opt.surface;
            axon_tree = morphData.traces_opt.axon;

            hem = string(metricsTable.hemisphere(iRow));

            nCells = nCells + 1;

            if hem == 'L'
                apical_tree.X = -1 * apical_tree.X;
                com_tree.X = -1 * com_tree.X;
                soma_tree.X = -1 * soma_tree.X;
            
                surface_tree.X = -1 * surface_tree.X;
                if ~isempty(axon_tree)
                    axon_tree.X = -1 * axon_tree.X;
                end
            end
        
            if plotApical
                if ~densityPlot
                    plot_tree(apical_tree,[0 0 1],[],[],[],'-3l');
                    %plot(apical_tree.X, apical_tree.Y, "Color", [0 0 1], 'LineWidth', plotWidth);
                end
                allX = [allX; apical_tree.X];
                allY = [allY; apical_tree.Y];

                cellX = [cellX; apical_tree.X];
                cellY = [cellY; apical_tree.Y];
            end
            if plotBasal
                if ~densityPlot
                    plot_tree(com_tree,[1 0 0],[],[],[],'-3l');
                    %plot(com_tree.X, com_tree.Y, "Color", [1 0 0], 'LineWidth', plotWidth);
                end
                allX = [allX; com_tree.X];
                allY = [allY; com_tree.Y];

                cellX = [cellX; com_tree.X];
                cellY = [cellY; com_tree.Y];
            end
            if ~densityPlot
                plot_tree(soma_tree,[],[],[],[],'-3l');
                if plotSurface && ~isempty(surface_tree)
                    plot_tree(surface_tree,[],[],[],[],'-3l');
                    %plot(surface_tree.X, surface_tree.Y, "Color", [0 0 0], 'LineWidth', plotWidth);

                end
                if plotAxon && ~isempty(axon_tree)
                    plot_tree(axon_tree,[1 0 1],[],[],[],'-3l');
                    %plot(axon_tree.X, axon_tree.Y, "Color", [1 0 1], 'LineWidth', plotWidth);

                    allX = [allX; axon_tree.X];
                    allY = [allY; axon_tree.Y];

                    cellX = [cellX; axon_tree.X];
                    cellY = [cellY; axon_tree.Y];
                end
            end

            cellDensity = histcounts2(cellY(:), cellX(:), xEdges, yEdges);

            % Unsure if the counts are in pixels or um, I think it's um.
            cellDensity = (cellDensity) / (binSize^2); 

            cellDensity = imgaussfilt(cellDensity, densitySigma);     

            densityMat = densityMat + cellDensity;
        end
    end

    scaleBarSize = 100; % 100um
    minX = min(allX);
    minY = min(allY);
    if densityPlot

        ptDensity = densityMat / nCells;
        
        ptDensity = imgaussfilt(ptDensity, densitySigma);       

        imagesc(xEdges, yEdges, ptDensity);

        hold on;

        xline(0, 'white', 'LineWidth', 1);
        yline(0, 'white', 'LineWidth', 1);

        imagesc(xEdges, yEdges, flipud(ptDensity));

        colormap(hot);

        midPix = int32(round(length(xEdges)/2));
        x1 = xEdges(1);
        x2 = x1 + scaleBarSize;
        y1 = yEdges(end) - 10;
        p1=plot([x1 x2],[y1 y1],'-','Color','white','LineWidth',1.5);

        colorbar()
        caxis([0, 0.6])
                
        axis equal;
        hold on;
    else        
        x1 = minX - scaleBarSize - 10;
        x2 = x1 + scaleBarSize;
        y1 = minY;
        p1=plot([x1 x2],[y1 y1],'-','Color','k','LineWidth',1.5);
    end
    
    
    hold off;
    box off; axis off;

    disp('Saving ...')
    %saveas(fig, savePath);
    plotDPI = 300;
    exportgraphics(fig, savePath, 'Resolution', plotDPI);    
    %export_fig(savePath, '-r300');    
    disp('Done.');
    
    close(fig);
    close all;
    clear fig;

end