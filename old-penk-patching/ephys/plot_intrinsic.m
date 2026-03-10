function [F2xRheo]=plot_intrinsic(data, plotType, filePath)
    %plot various ephys traces
    %data=input structure
    %col=color for traces
    %ty=what type of trace (e.g. IV, Rheobase, Passive,Sag, Ramp)
    
    sr = 20000;
    srF = 20;
    plotDPI = 300;
    
    if plotType==1
%         step=5;
%         if ~isempty(data)
%              ov_min = min(min(data.traces(:,1:step:end))); 
%              ov_max = max(max(data.traces(:,1:step:end))); 
%         end

        % First add all sub thresh
        subThreshIndexes = data.spikecount == 0;
        firstSpikeIndex = find(data.spikecount, 1, 'first');
        [~, maxSpikeIndex] = max(data.spikecount);

        subTraces = data.traces(:, subThreshIndexes);
        threshTrace = data.traces(:, firstSpikeIndex);
        maxTrace = data.traces(:, maxSpikeIndex);
        lastTrace = data.traces(:, end);

        threshStim = data.stimvec(firstSpikeIndex);
        maxStim = data.stimvec(maxSpikeIndex);
        endStim = data.stimvec(end);

        plotTraces = subTraces;
        plotTraces = [plotTraces threshTrace];
        plotTraces = [plotTraces maxTrace];
        plotTraces = [plotTraces lastTrace];

        ov_min = min(min(plotTraces)); 
        ov_max = max(max(plotTraces)); 
    
        %% Plotting IV
        %fig=figure;set(fig1, 'Position', [200, -200, 800, 800]);
        fig=figure('visible','off'); 
        set(gcf,'color','w');
   
        if ~isempty(data)
            %Plot only every second trace atm for display
            plot(subTraces, 'Color','k','LineWidth', 1);
            set(gca,'ColorOrderIndex',1)
            hold on;
            
            plot(threshTrace, 'LineWidth', 1);
            plot(maxTrace, 'LineWidth', 1);
            %plot(lastTrace, 'LineWidth', 1);
           
            textStart = -150*srF;
            text(textStart, data.RMP, [num2str(data.RMP), 'mV'], 'FontSize', 9);
            threshTraceY = median(threshTrace);
            text(textStart, threshTraceY, [num2str(threshStim), ''], 'FontSize', 9);
            maxTraceY = median(maxTrace);
            if maxTraceY - threshTraceY < 10
                maxTraceY = maxTraceY + 10;
            end
            text(textStart, maxTraceY, [num2str(maxStim), ''], 'FontSize', 9);
            %text(textStart, median(lastTrace), [num2str(endStim), ''], 'FontSize', 9);

            %Scale bar
            scale_x = 0.1 * sr;
            scale_y = 10;
            pos_x = 0;
            pos_y = data.RMP - scale_y * 1.2;
            drawScaleBar(scale_x, scale_y, pos_x, pos_y);

            hold off;
            ylim([ov_min-10 ov_max]);
            xlim([textStart-10 length(threshTrace)+10]);
            set(gca,'box','off');axis off;
        else
            plot(1,1);set(gca,'box','off');axis off;
        end


        %saveas(fig, filePath);
        exportgraphics(fig, filePath, 'Resolution', plotDPI);
        close(fig);
        clear fig;
        
    elseif plotType==2 %Rheobase
        trace_1xRheo=[];
        trace_2xRheo=[];
        if isempty(data)==0
            %find Rheobase 
            rheo = data.rheo;
            
            idx_1xRheo = find(data.stimvec==1*data.rheo);
            trace_1xRheo = data.traces(:, idx_1xRheo);

            if ~isempty(find(data.spikecount>=2))
                md=find(data.spikecount>=2);
                trace_2ap = data.traces(:,md(1));
                ap2 = data.stimvec(md(1));
            else
                md=find(data.spikecount>=1);
                trace_2ap=data.traces(:,md(1));
            end

            % Get 2x the rheobase, or the last.
            idx_2xRheo=find(data.stimvec==round(2*ap2,2,'significant'));
            has2xRheo = ~isempty(idx_2xRheo);
            if ~has2xRheo
                idx_2xRheo = size(data.traces, 2);
            end

            trace_2xRheo=data.traces(:,idx_2xRheo);
            
        end

        data.stimvec
       
     
        %fig=figure;set(fig1, 'Position', [200, -200, 800, 800]);
        fig=figure('visible','off'); 
        set(gcf,'color','w');

       
        plot(trace_1xRheo,'Color', 'k','LineWidth',1); 
        set(gca,'ColorOrderIndex',1)
        hold on; 
        
        plot(data.traces(:, 1:idx_1xRheo - 1),'Color','k','LineWidth',1);
        %plot(trace_2xRheo,'Color','r','LineWidth',1);

        textStart = 0; %-100*srF;

        max1xRheoTrace = median(trace_1xRheo);
%         max2xRheoTrace = median(trace_2xRheo);
%         if max2xRheoTrace - max1xRheoTrace < 10*srF
%             max2xRheoTrace = max2xRheoTrace + 10*srF;
%         end

        text(textStart, max1xRheoTrace, [num2str(data.rheo), ''], 'FontSize', 9);
        %text(textStart, max2xRheoTrace, [num2str(data.stimvec(idx_2xRheo)), ''], 'FontSize', 9);
       
        %Scale bar

        scale_x = 0.1 * sr;
        scale_y = 10;
        pos_x = 0;
        pos_y = max1xRheoTrace + 10;
        drawScaleBar(scale_x, scale_y, pos_x, pos_y);

        set(gca,'box','off');axis off;
%         if has2xRheo
%             legend({'Rheobase', '2x Rheobase'}, 'location','best');
%         else
%             legend({'Rheobase', 'Max'}, 'location','best');
%         end


        %saveas(fig, filePath);
        exportgraphics(fig, filePath, 'Resolution', plotDPI);
        close(fig);
        clear fig;


    else
        error()
    end

end