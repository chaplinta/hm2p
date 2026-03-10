function statsTable = sumTableNum(dataTable, ...
                                 statsSumCSVFile, ...
                                 dirs)

    if nargin < 2
        statsSumCSVFile = '';
    end


    colTypes = varfun(@class, dataTable, 'OutputFormat', 'cell');

    penkPosIndexes = startsWith(string(dataTable.cell_type), 'penkpos'); 
    penkNegIndexes = startsWith(string(dataTable.cell_type), 'penkneg'); 

    
    statsTable = [];
    iStat = 0;

    metricMat = [];
    metricMatNames = {};

    for iCol=1:width(dataTable)

        metricName = string(dataTable.Properties.VariableNames{iCol});
        nonMetrics = {'cell_index', 'animal_id', 'cell_slice_id', 'cell_type', ...
                      'layer', 'dob', 'has_ephys', 'has_ephys_complete', ...
                      'has_morph', 'has_morph_good', 'active_spike_index', 'active_time_index'};

        if strcmp('double', string(colTypes(iCol))) && ...
            ~any(strcmp(nonMetrics, metricName))

            colData = dataTable.(iCol);
            hasDataIndexes = ~isnan(colData);
            %colData = colData(hasDataIndexes);

            colDataPos = dataTable.(metricName)(penkPosIndexes & hasDataIndexes);
            colDataNeg = dataTable.(metricName)(penkNegIndexes & hasDataIndexes);
    
            iStat = iStat + 1;
           
            statsTable(iStat).metric = metricName;

%             statsTable(iStat).n = length(colData);
%             statsTable(iStat).mean = mean(colData);
%             statsTable(iStat).median = median(colData);
%             statsTable(iStat).std = std(colData, 0, 1);
%             statsTable(iStat).sem = statsTable(iStat).std / ...
%                         sqrt(statsTable(iStat).n);
%             statsTable(iStat).min = min(colData);
%             statsTable(iStat).max = max(colData);
%             prc = prctile(colData, [2.5, 97.5]);
%             %statsTable(iStat).prc95Lo = prc(1);
%             %statsTable(iStat).prc95Hi = prc(2);

            statsTable(iStat).penkpos_n = length(colDataPos);
            statsTable(iStat).penkpos_mean = mean(colDataPos);
            statsTable(iStat).penkpos_median = median(colDataPos);
            statsTable(iStat).penkpos_std = std(colDataPos, 0, 1);
            statsTable(iStat).penkpos_sem = statsTable(iStat).penkpos_std / ...
                        sqrt(statsTable(iStat).penkpos_n);
            statsTable(iStat).penkpos_min = min(colDataPos);
            statsTable(iStat).penkpos_max = max(colDataPos);

            statsTable(iStat).penkneg_n = length(colDataNeg);
            statsTable(iStat).penkneg_mean = mean(colDataNeg);
            statsTable(iStat).penkneg_median = median(colDataNeg);
            statsTable(iStat).penkneg_std = std(colDataNeg, 0, 1);
            statsTable(iStat).penkneg_sem = statsTable(iStat).penkneg_std / ...
                        sqrt(statsTable(iStat).penkneg_n);
            statsTable(iStat).penkneg_min = min(colDataNeg);
            statsTable(iStat).penkneg_max = max(colDataNeg);

            statsTable(iStat).penkposneg_p = NaN;
            if length(colDataPos) > 0 && length(colDataNeg) > 0
                statsTable(iStat).penkposneg_p = ranksum(colDataPos, colDataNeg);     
            end

            plot_data = [colDataPos; colDataNeg];
            plot_types_pos = string(dataTable.cell_type(penkPosIndexes & hasDataIndexes));
            plot_types_neg = string(dataTable.cell_type(penkNegIndexes & hasDataIndexes));
            cell_ids_pos = dataTable.cell_index(penkPosIndexes & hasDataIndexes);
            cell_ids_neg = dataTable.cell_index(penkNegIndexes & hasDataIndexes);
            % Fuck me cannot figure out how to do this properly in matlab.
            plot_types = {};
            for i=1:size(plot_data, 1)
                if i <= length(colDataPos)
                    plot_types{i} = 'Penk positive';
                else
                    plot_types{i} = 'Penk negative';
                end
            end

           
            fig = figure('visible','off');
            %set(fig, 'Position', [100 100 100 500])
            vp = violinplot(plot_data, plot_types, ...
                'GroupOrder', {'Penk positive', 'Penk negative'});

            for iVP=1:length(vp)
                x = vp(iVP).ScatterPlot.XData;
                y = vp(iVP).ScatterPlot.YData;
                % displacement so the text does not overlay the data points
                dx = 0.01*range(x); dy = 0.01*range(y); 
                if iVP == 1
                    cell_ids = cell_ids_pos;
                else
                    cell_ids = cell_ids_neg;
                end
                text(x+dx, y+dy, ...
                     string(cell_ids), ...
                     'FontSize', 6);
            end
            
            titleStr = metricName.replace('_', '\_');
            p = statsTable(iStat).penkposneg_p;
            if ~isnan(p)
                titleStr = strcat(titleStr, sprintf(" p=%.3f", p));
            end
            title(titleStr);
            xlabel('');
            saveas(fig, fullfile(dirs.statSumViolinDir, strcat(metricName, ".png")));
            close(fig);                          

           
            % Box plots

            % Plot just outliers
            fig = figure('visible','off');
            grpIndex = grp2idx(plot_types);
            h = boxchart(grpIndex, ...
                         plot_data);

            titleStr = metricName.replace('_', '\_');
            p = statsTable(iStat).penkposneg_p;
            if ~isnan(p)
                titleStr = strcat(titleStr, sprintf(" p=%.3f", p));
            end
            title(titleStr);
            xlabel('');

            set(gca, 'TickDir','out');
            set(gca, 'box','off');
            set(gca, 'LineWidth',2);
            set(h, {'linew'}, {2})
            set(gca,"XTick", unique(grpIndex), "XTickLabel", ...
                categorical(plot_types), 'fontsize', 12);
            
            saveas(fig, fullfile(dirs.statSumBoxDir, strcat(metricName, ".png")));
            close(fig); 

            % Plot all points
            fig = figure('visible','off');
            grpIndex = grp2idx(plot_types);
            h = boxchart(grpIndex, ...
                         plot_data, "MarkerStyle","none");

            titleStr = metricName.replace('_', '\_');
            p = statsTable(iStat).penkposneg_p;
            if ~isnan(p)
                titleStr = strcat(titleStr, sprintf(" p=%.3f", p));
            end
            title(titleStr);
            xlabel('');

            set(gca, 'TickDir','out');
            set(gca, 'box','off');
            set(gca, 'LineWidth',2);
            set(h, {'linew'}, {2})
            set(gca,"XTick", unique(grpIndex), "XTickLabel", ...
                categorical(plot_types), 'fontsize', 12);

            hold on;
            for iGrp=1:max(unique(grpIndex))
                xPos = ones(sum(grpIndex==iGrp),1) + iGrp-1;
                jitter = 0.2;
                rng(4);
                xPos = xPos + (rand(size(xPos)) - 0.5) * jitter;
                yPos = plot_data(grpIndex == iGrp);
                hs = scatter(xPos, ...
                             yPos, ...
                             72, ...
                             'k', "filled"); %,'jitter','on','JitterAmount',0.1);
                hs.MarkerFaceAlpha = 0.75;

%                     dx = 0.03*range(xPos); 
%                     dy = 0.03*range(yPos); 
%                     if iGrp == 1
%                         cell_ids = cell_ids_pos;
%                     else
%                         cell_ids = cell_ids_neg;
%                     end
%                     text(xPos+dx, yPos+dy, ...
%                          string(cell_ids), ...
%                          'FontSize', 10);

            end
            hold off;

            saveas(fig, fullfile(dirs.statSumBoxPtsDir, strcat(metricName, ".png")));
            close(fig); 
            
                           

%             % Histograms
%             nBins = 10;
%             fig = figure('visible','off');
%             histogram([colDataPos; colDataNeg], nBins);
%             xlabel(metricName.replace('_', '\_'));
%             ylabel('# cells');
%             saveas(fig, fullfile(dirs.statSumHistDir, strcat(metricName, ".png")));
%             close(fig);              
            
        end
    end

    statsTable = struct2table(statsTable);

    if ~isempty(statsSumCSVFile)
        writetable(statsTable, statsSumCSVFile, 'Delimiter', ',')  
    end

end