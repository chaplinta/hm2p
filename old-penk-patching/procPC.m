clear all;

loadPreprocssed = false;
loadPreprocssedAnswer = questdlg('Load preprocssed data or process raw data gain?', ...
	'Processing raw data', ...
	'Load preprocessed', 'Process again', 'Cancel', 'Load preprocessed');
switch loadPreprocssedAnswer
    case 'Load preprocessed'
        disp('Loading preprocessed data (where available).')
        loadPreprocssed = true;
    case 'Process again'
        disp('Processing raw data again.')
        loadPreprocssed = false;
    case 'Cancel'
        disp('Quiting before processing anything');
        return
end

plotDPI = 300;

dirs = loadPCDirs();

animalsTable = readtable(dirs.metaAnimalsPath);
cellsTable = readtable(dirs.metaCellsPath);

refData = join(cellsTable, animalsTable, 'Keys', {'animal_id'});

nCells = height(cellsTable);

metricsTable = [];
iMetric = 0;
for iCell = 1:nCells

    ephysId = string(refData.ephys_id(iCell));
    has_ephys = ~(ephysId == "");
    has_good_morph = refData.has_morph(iCell) && refData.good_morph(iCell);
   
    % Check to see if there are cells in the other hemisphere, if so then
    % the tracing is in a subfolder.
    cells_both_hem = false;
    hem = string(refData.hemisphere(iCell));
    hem_contra = '';
    if strcmp(hem, 'R')
        hem_contra = 'L';
    elseif strcmp(hem, 'L')
        hem_contra = 'R';
    end
    other_hem_cells = string(refData.animal_id) == string(refData.animal_id(iCell)) & ...
                      string(refData.slice_id) == string(refData.slice_id(iCell)) & ...
                      string(refData.hemisphere) == hem_contra;

    cells_both_hem = ~strcmp(hem_contra, '') & sum(other_hem_cells) > 0;

    dateConfcalStr = string(refData.date_confocal(iCell));
    dataPatchStr   = string(refData.date_slice(iCell));

    sliceFolder = strcat(string(refData.date_slice(iCell)), '_', ...
                  string(refData.slice_id(iCell)));
    cellFolder  = strcat('Cell', string(refData.cell_slice_id(iCell)));

    hemPath = fullfile(dirs.morphDir, dateConfcalStr, sliceFolder);
    tracingPath = fullfile(hemPath, dirs.tracingSubDir, cellFolder);

    % Bit of a hack here to handle inconsistencies in sub folders and
    % hemispheres.
    % If there are cells in both hemispheres then there is a sub folder for
    % each hemisphere. 
    % If there are only cells in one hemisphere, then sometimes there is 
    % no sub folder for the hemisphere, sometimes there is.
    % Therefore use the subfolder if there are cells on both hemispheres,
    % or if the sub folder exists (i.e. default tracingPath doesn't exist).
    if cells_both_hem || ~exist(tracingPath, 'dir')
            
        % Add the hemisphere subfolder.
        hemFolderName = strcat(sliceFolder, '_', hem);
        hemPath = fullfile(dirs.morphDir, dateConfcalStr, sliceFolder, ...
                           hemFolderName);
        tracingPath = fullfile(hemPath, dirs.tracingSubDir, cellFolder);

        
    end

    cellName = strcat(sprintf( '%03d', refData.cell_index(iCell)), "-", ...
                      string(refData.animal_id(iCell)), "-", ...
                      string(refData.slice_id(iCell)), "-", ...
                      string(refData.cell_slice_id(iCell)));

    fprintf('%s %s %s %s\n', ...
            cellName, ...
            fillmissing(ephysId, 'constant', "N/A"), ...
            fillmissing(dateConfcalStr, 'constant', "N/A"), ...
            fillmissing(dataPatchStr, 'constant', "N/A"));

    cellNameDir = fullfile(dirs.processedDir, cellName);
    [success, msg, msgid] = mkdir(cellNameDir);
    cellMorphDataPath = fullfile(cellNameDir, 'morph_data.mat');
    cellEPhysDataPath = fullfile(cellNameDir, 'ephys_data.mat');


    iMetric = iMetric + 1;
    metricsTable(iMetric).cell_index = int32(refData.cell_index(iCell));
    metricsTable(iMetric).animal_id = string(refData.animal_id(iCell));
    metricsTable(iMetric).slice_id = string(refData.slice_id(iCell));
    metricsTable(iMetric).cell_slice_id = string(refData.cell_slice_id(iCell));
    metricsTable(iMetric).hemisphere = hem;
    metricsTable(iMetric).cell_type = string(refData.cell_type(iCell));
    metricsTable(iMetric).depth_slice = double(refData.depth_slice(iCell));
    %metricsTable(iMetric).depth_pial = double(refData.depth_pial(iCell));
    metricsTable(iMetric).depth_pial = NaN;
    metricsTable(iMetric).area = string(refData.area(iCell));
    metricsTable(iMetric).layer = string(refData.layer(iCell));
    metricsTable(iMetric).has_ephys = false;
    metricsTable(iMetric).has_ephys_complete = false;    
    metricsTable(iMetric).has_morph = refData.has_morph(iCell);
    metricsTable(iMetric).has_morph_good = false;

    metricsTable(iMetric).proc_name = cellName;
    

    metricsTable(iMetric).ephys_passive_RMP = NaN;
    metricsTable(iMetric).ephys_passive_rhreo = NaN;
    metricsTable(iMetric).ephys_passive_rin = NaN;
    metricsTable(iMetric).ephys_passive_tau = NaN;
    metricsTable(iMetric).ephys_passive_incap = NaN;
    metricsTable(iMetric).ephys_passive_sag = NaN;
   
    metricsTable(iMetric).ephys_passive_maxsp = NaN;
    metricsTable(iMetric).ephys_active_minVm = NaN; 
    metricsTable(iMetric).ephys_active_peakVm = NaN;     
    %metricsTable(iMetric).ephys_active_initVm = NaN;     
    %metricsTable(iMetric).ephys_active_initVmBySlope = NaN;
    metricsTable(iMetric).ephys_active_maxVmSlope = NaN;   
    metricsTable(iMetric).ephys_active_halfVm = NaN;   
    metricsTable(iMetric).ephys_active_amplitude = NaN;   
    metricsTable(iMetric).ephys_active_maxAHP = NaN;       
    %metricsTable(iMetric).ephys_active_DAHPMag = NaN;    
    %metricsTable(iMetric).ephys_active_initTime = NaN;    
    %metricsTable(iMetric).ephys_active_riseTime = NaN;     
    %metricsTable(iMetric).ephys_active_fallTime = NaN;    
    %metricsTable(iMetric).ephys_active_minTime = NaN;     
    %metricsTable(iMetric).ephys_active_baseWidth = NaN;   
    metricsTable(iMetric).ephys_active_halfWidth = NaN;    
    %metricsTable(iMetric).ephys_active_fixVWidth = NaN; 
    % No ephys prefix so it's not picked up PCA.
    metricsTable(iMetric).active_spike_index = NaN;        
    metricsTable(iMetric).active_time_index = NaN;

    metricsTable(iMetric).morph_api_len = NaN;
    metricsTable(iMetric).morph_api_max_plen = NaN;
    metricsTable(iMetric).morph_api_bpoints = NaN;
    metricsTable(iMetric).morph_api_mpeucl = NaN;
    metricsTable(iMetric).morph_api_maxbo = NaN;
    %metricsTable(iMetric).morph_api_mangleB = NaN;
    metricsTable(iMetric).morph_api_mblen = NaN;
    metricsTable(iMetric).morph_api_mplen = NaN;
    metricsTable(iMetric).morph_api_mbo = NaN;
    metricsTable(iMetric).morph_api_width = NaN;
    metricsTable(iMetric).morph_api_height = NaN;
    metricsTable(iMetric).morph_api_depth = NaN;
    metricsTable(iMetric).morph_api_wh = NaN;
    metricsTable(iMetric).morph_api_wd = NaN;
    metricsTable(iMetric).morph_api_shlpeakcr = NaN;
    metricsTable(iMetric).morph_api_shlpeakcrdist = NaN;
    metricsTable(iMetric).morph_api_ext_super = NaN;
    metricsTable(iMetric).morph_api_ext_deep = NaN;
    %metricsTable(iMetric).morph_api_wz = NaN;
    %metricsTable(iMetric).morph_api_chullx = NaN;
    %metricsTable(iMetric).morph_api_chully = NaN;
    %metricsTable(iMetric).morph_api_chullz = NaN;

    metricsTable(iMetric).morph_bas_len = NaN;
    metricsTable(iMetric).morph_bas_max_plen = NaN;
    metricsTable(iMetric).morph_bas_bpoints = NaN;
    metricsTable(iMetric).morph_bas_mpeucl = NaN;
    metricsTable(iMetric).morph_bas_maxbo = NaN;
    %metricsTable(iMetric).morph_bas_mangleB = NaN;
    metricsTable(iMetric).morph_bas_mblen = NaN;
    metricsTable(iMetric).morph_bas_mplen = NaN;
    metricsTable(iMetric).morph_bas_mbo = NaN;
    metricsTable(iMetric).morph_bas_width = NaN;
    metricsTable(iMetric).morph_bas_height = NaN;
    metricsTable(iMetric).morph_bas_depth = NaN;
    metricsTable(iMetric).morph_bas_wh = NaN;    
    metricsTable(iMetric).morph_bas_wd = NaN;
    metricsTable(iMetric).morph_bas_ntrees = NaN;
    metricsTable(iMetric).morph_bas_shlpeakcr = NaN;
    metricsTable(iMetric).morph_bas_shlpeakcrdist = NaN;
    metricsTable(iMetric).morph_bas_ext_super = NaN;
    metricsTable(iMetric).morph_bas_ext_deep = NaN;    
    %metricsTable(iMetric).morph_bas_wz = NaN;
    %metricsTable(iMetric).morph_bas_chullx = NaN;
    %metricsTable(iMetric).morph_bas_chully = NaN;
    %metricsTable(iMetric).morph_bas_chullz = NaN;

    if has_ephys

        if loadPreprocssed && isfile(cellEPhysDataPath)    
            ephysData = load(cellEPhysDataPath, 'ephysData');
            
            % Somehow this ends up in a struct with the same name.
            ephysData = ephysData.ephysData;
        else
            ephysFolder = fullfile(dirs.ephysDir, dataPatchStr, ephysId);
    
            intrList = {'IV', 'Passive', 'Rheobase', 'Passive', 'Sag', 'Ramp'};
            h5List = dir([char(ephysFolder) '\*.h5']);
            intrIndexes = find(contains({h5List(:).name}, intrList) == 1);
        
            [IV Rheobase Passive Sag Ramp] = ephys_intr(h5List, ...
                                                        intrIndexes, ...
                                                        ephysFolder);
            clear ephysData;
            ephysData.IV = IV;
            ephysData.Rheobase = Rheobase;
            ephysData.Passive = Passive;
            ephysData.Sag = Sag;
            ephysData.Ramp = Ramp;

            % Get the active spikes from the rheobase test, when there was at
            % least 1 spike (currently not used).
            md1 = find(ephysData.Rheobase.spikecount >= 1);
            trace1 = ephysData.Rheobase.traces(:, md1(1));
    
            % Get the active spikes from the rheobase test, when there were at
            % least 2 spikes.
            md2 = find(ephysData.Rheobase.spikecount >= 2);
            trace2 = ephysData.Rheobase.traces(:, md2(1));
    
            % Use the second spike for active stats.
            active_spike_index = 2;
            [active_p] = sp_parameters_pandora(trace2, active_spike_index);

            ephysData.active.params = active_p;
            ephysData.active.trace = trace2;
        
            save(cellEPhysDataPath, 'ephysData');
        end
        
        metricsTable(iMetric).has_ephys = true;
        metricsTable(iMetric).has_ephys_complete = true;
        if ~isempty(ephysData.IV)
            metricsTable(iMetric).ephys_passive_RMP = ephysData.IV.RMP;
        else
            metricsTable(iMetric).ephys_passive_RMP = ephysData.Rheobase.RMP;
        end
        metricsTable(iMetric).ephys_passive_rhreo = ephysData.Rheobase.rheo;
        metricsTable(iMetric).ephys_passive_rin = ephysData.Passive.Rin;        
        metricsTable(iMetric).ephys_passive_tau = ephysData.Passive.tau(1);
        metricsTable(iMetric).ephys_passive_incap = metricsTable(iMetric).ephys_passive_tau / metricsTable(iMetric).ephys_passive_rin;
        
        if ~isempty(ephysData.Sag)
            metricsTable(iMetric).ephys_passive_sag = ephysData.Sag.sagratio(1);
        else
            metricsTable(iMetric).has_ephys_complete = false;
        end

        

        % Stim is 1 second.
        if ~isempty(ephysData.IV)
            metricsTable(iMetric).ephys_passive_maxsp = double(max(ephysData.IV.spikecount));
        else
            metricsTable(iMetric).has_ephys_complete = true;
        end

        % Only use the following active metrics:
        % ephys_active_maxsp
        % MinVm       
        % PeakVm     
        % MaxVmSlope  
        % HalfVm      
        % Amplitude    
        % MaxAHP              
        % HalfWidth  
     
        % Sample rate is assumed to be 10kHz when it's actually 20kHz, so
        % divide or multiply by 2 as needed.
        metricsTable(iMetric).ephys_active_minVm = ephysData.active.params(1);  
        metricsTable(iMetric).ephys_active_peakVm = ephysData.active.params(2);       
        %metricsTable(iMetric).ephys_active_initVm = ephysData.active.params(3);      
        %metricsTable(iMetric).ephys_active_initVmBySlope = ephysData.active.params(4);
        metricsTable(iMetric).ephys_active_maxVmSlope = ephysData.active.params(5) * 2.0;   
        metricsTable(iMetric).ephys_active_halfVm = ephysData.active.params(6);      
        metricsTable(iMetric).ephys_active_amplitude = ephysData.active.params(7);    
        metricsTable(iMetric).ephys_active_maxAHP = ephysData.active.params(8);       
        %metricsTable(iMetric).ephys_active_DAHPMag = ephysData.active.params(9);     
        %metricsTable(iMetric).ephys_active_initTime = ephysData.active.params(10);     
        %metricsTable(iMetric).ephys_active_riseTime = ephysData.active.params(11);     
        %metricsTable(iMetric).ephys_active_fallTime = ephysData.active.params(12);     
        %metricsTable(iMetric).ephys_active_minTime = ephysData.active.params(13);      
        %metricsTable(iMetric).ephys_active_baseWidth = ephysData.active.params(14);    
        metricsTable(iMetric).ephys_active_halfWidth = ephysData.active.params(15) / 2.0;    
        %metricsTable(iMetric).ephys_active_fixVWidth = ephysData.active.params(16);
        % No ephys prefix so it's not picked up PCA.
        metricsTable(iMetric).active_spike_index = ephysData.active.params(17);        
        % For some reason time is off by a factor of 10, nfi why.
        % (i.e. it's not sampling rate (should be 3x in that case))
        % No ephys prefix so it's not picked up PCA.
        metricsTable(iMetric).active_time_index = int32(ephysData.active.params(18) * 10);


      

    end

    if has_good_morph

        [swcList] = getTracingFiles(tracingPath);

        if loadPreprocssed && isfile(cellMorphDataPath)   
            morphData = load(cellMorphDataPath, 'morphData');
            % Somehow this ends up in a struct with the same name.
            morphData = morphData.morphData;
        else            
            morphData = morphology_readout(swcList, false, cellMorphDataPath);    
            save(cellMorphDataPath, 'morphData');
        end

        metricsTable(iMetric).has_morph_good = true;

        metricsTable(iMetric).depth_pial = morphData.surface_stats.dist_soma;
        %metricsTable(iMetric).depth_slice = morphData.soma_stats.mz;

        % Don't use the following metrics
        % morph_api_mangleB
        % morph_api_wz - z is not reliable with confocal
        % morph_api_chullx
        % morph_api_chully
        % morph_api_chullz


        % Fields from Weiler et all that need to be added.
%         Radial Dis.max: Maximal radial distance from soma

        metricsTable(iMetric).morph_api_len = morphData.apical_stats.gstats.len; % Total length: Total length of tree
        metricsTable(iMetric).morph_api_max_plen = morphData.apical_stats.gstats.max_plen; % Path Length max: Maximal path length from soma
        metricsTable(iMetric).morph_api_bpoints = morphData.apical_stats.gstats.bpoints; % Branch Points: Number of branch points
        metricsTable(iMetric).morph_api_mpeucl = morphData.apical_stats.gstats.mpeucl; % Path length/eucl length NOTE: not in Simon's paper??
        metricsTable(iMetric).morph_api_maxbo = morphData.apical_stats.gstats.maxbo; % Branch Order max: Maximal branch order        
        metricsTable(iMetric).morph_api_mblen = morphData.apical_stats.gstats.mblen; % Branch Length mean: Mean branch length
        metricsTable(iMetric).morph_api_mplen = morphData.apical_stats.gstats.mplen; % Path Length mean: Mean branch length
        metricsTable(iMetric).morph_api_mbo = morphData.apical_stats.gstats.mbo; % Mean branch order. NOTE: not in Simon's paper??
        metricsTable(iMetric).morph_api_width = morphData.apical_stats.gstats.width; % Width: maximal horizontal span
        metricsTable(iMetric).morph_api_height = morphData.apical_stats.gstats.height; % Height: maximal vertical span
        metricsTable(iMetric).morph_api_depth = morphData.apical_stats.gstats.depth; % Depth: maximal z span
        metricsTable(iMetric).morph_api_wh = morphData.apical_stats.gstats.wh; % Width / Height of tree
        metricsTable(iMetric).morph_api_wd = morphData.apical_stats.gstats.wd; % Width / Depth of tree

        [shlpeakcr, index] = max(morphData.apical_stats.dstats.sholl{1});
        metricsTable(iMetric).morph_api_shlpeakcr = shlpeakcr;
        metricsTable(iMetric).morph_api_shlpeakcrdist = morphData.apical_stats.dsholl(index);

        % Find closest and furtherest dendrite from the surface.
        surfPts = [morphData.traces_opt.surface.X morphData.traces_opt.surface.Y];
        dendApiPts = [morphData.traces{1}.X morphData.traces{1}.Y];
        surfDistsApi = getSurfDist(surfPts, dendApiPts);
        metricsTable(iMetric).morph_api_ext_super = surfDistsApi(1);
        metricsTable(iMetric).morph_api_ext_deep = surfDistsApi(2);
 
        metricsTable(iMetric).morph_bas_len = morphData.basal_stats.gstats.len;
        metricsTable(iMetric).morph_bas_max_plen = morphData.basal_stats.gstats.max_plen;
        metricsTable(iMetric).morph_bas_bpoints = morphData.basal_stats.gstats.bpoints;
        metricsTable(iMetric).morph_bas_mpeucl = morphData.basal_stats.gstats.mpeucl;
        metricsTable(iMetric).morph_bas_maxbo = morphData.basal_stats.gstats.maxbo;
        metricsTable(iMetric).morph_bas_mblen = morphData.basal_stats.gstats.mblen;
        metricsTable(iMetric).morph_bas_mplen = morphData.basal_stats.gstats.mplen;
        metricsTable(iMetric).morph_bas_mbo = morphData.basal_stats.gstats.mbo;        
        metricsTable(iMetric).morph_bas_width = morphData.basal_stats.gstats.width;
        metricsTable(iMetric).morph_bas_height = morphData.basal_stats.gstats.height;
        metricsTable(iMetric).morph_bas_depth = morphData.basal_stats.gstats.depth; 
        metricsTable(iMetric).morph_bas_wh = morphData.basal_stats.gstats.wh;
        metricsTable(iMetric).morph_bas_wd = morphData.basal_stats.gstats.wd;

        metricsTable(iMetric).morph_bas_ntrees = morphData.basal_stats.gstats.basaltrees;
        [shlpeakcr, index] = max(morphData.basal_stats.dstats.sholl{1});
        metricsTable(iMetric).morph_bas_shlpeakcr = shlpeakcr;
        metricsTable(iMetric).morph_bas_shlpeakcrdist = morphData.basal_stats.dsholl(index);

        % Find closest and furtherest dendrite from the surface.
        surfPts = [morphData.traces_opt.surface.X morphData.traces_opt.surface.Y];
        dendBasPts = [morphData.traces{2}.X morphData.traces{2}.Y];
        surfDistsBas = getSurfDist(surfPts, dendBasPts);
        metricsTable(iMetric).morph_bas_ext_super = surfDistsBas(1);
        metricsTable(iMetric).morph_bas_ext_deep = surfDistsBas(2);

        imageStackPath = fullfile(hemPath, "corrected.tif");
        if ~isfile(imageStackPath)
            error("Could not find image stack %s", imageStackPath);
        end
        
        apical_tree = morphData.traces{1};
        com_tree = morphData.traces{2};
        soma_tree = morphData.traces{3};
        
        surface_tree = morphData.traces_opt.surface;
        axon_tree = morphData.traces_opt.axon;

        morphFigPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-01-plot.png"));

        if ~isfile(morphFigPath)
          
            fig=figure('visible','off');
            mon_pos=get(0,'MonitorPositions');
            set(gcf,'color','w', 'menubar','figure', 'position', [mon_pos(1,3)-1200 2 500 500]); % [left, bottom, width, height]
            HP=plot_tree(apical_tree,[0 0 1],[],[],[],'-3l');
            hold on;         
            plot_tree(com_tree,[1 0 0],[],[],[],'-3l');
            plot_tree(soma_tree,[],[],[],[],'-3l');
            if ~isempty(surface_tree)
                plot_tree(surface_tree,[],[],[],[],'-3l');
            end
            if ~isempty(axon_tree)
                plot_tree(axon_tree,[1 0 1],[],[],[],'-3l');
            end

            scaleBarSize = 100; % 100um
            minX = min([apical_tree.X; com_tree.X]);
            minY = min([apical_tree.Y; com_tree.Y]);
            x1 = minX - scaleBarSize - 10;
            x2 = x1 + scaleBarSize;
            y1 = minY - 50;
            p1=plot([x1 x2],[y1 y1],'-','Color','k','LineWidth',1.5);

            hold off;
            box off; axis off;
            
            %saveas(fig, morphFigPath);
            exportgraphics(fig, morphFigPath, 'Resolution', plotDPI);
            close(fig); 
        end


        % Load the tiff stack and make some images.            
        stackCellMeanPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-02-mean.png"));
        stackCellMeanCropPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-03-mean-crop.png"));
        stackCellMaxPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-04-max.png"));
        stackCellMaxCropPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-05-max-crop.png"));
        stackCellGIFPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-06-stack.gif"));
        stackCellGIFCropPath = fullfile(dirs.morphPlotDir, strcat(cellName, "-07-stack-crop.gif"));

        if  ~(isfile(stackCellMeanPath) && ...
             isfile(stackCellMaxPath) && ...
             isfile(stackCellGIFPath))

            tiffInfo = imfinfo(imageStackPath);
            imageWidth = tiffInfo(1).Width;
            imageHeight = tiffInfo(1).Height;
            imagePixXRes = 1.0 / tiffInfo(1).XResolution;
            imagePixYRes = 1.0 / tiffInfo(1).YResolution; 
            nFrames = numel(tiffInfo);

            if nFrames == 1
                % If the file is bigger than 4GB it fails to load properly
                % here, so just skip for now,
                continue
            end
            % Description is frm fiji and has the spacing.
            imageDesc = tiffInfo(1).ImageDescription; 
            spacingStr = regexp(imageDesc, 'spacing=(\d*\.?\d*)', 'tokens');
            spacingStr = spacingStr{1}{1};
            imagePixZRes = str2double(spacingStr);

            xyPad = 10;
    
            frameMin = ceil((min([com_tree.Z; apical_tree.Z]) + ...
                              morphData.soma_stats.mz) / imagePixZRes);
            frameMax = floor((max([com_tree.Z; apical_tree.Z]) + ...
                              morphData.soma_stats.mz) / imagePixZRes);

            xMin = ceil((min([com_tree.X; apical_tree.X]) + ...
                              morphData.soma_stats.mx) / imagePixXRes) - xyPad;
            xMax = floor((max([com_tree.X; apical_tree.X]) + ...
                              morphData.soma_stats.mx) / imagePixXRes) + xyPad;

            yMin = ceil((min(-[com_tree.Y; apical_tree.Y]) + ...
                              morphData.soma_stats.my) / imagePixYRes) - xyPad;
            yMax = floor((max(-[com_tree.Y; apical_tree.Y]) + ...
                              morphData.soma_stats.my) / imagePixYRes) + xyPad;

            imgSomaPadX = ceil(morphData.soma_stats.mx / imagePixXRes);
            imgSomaPadY = ceil(morphData.soma_stats.my / imagePixYRes);
            imgSomaPadXDir = 'post';
            imgSomaPadYDir = 'pre';
            if imgSomaPadX < 0
                imgSomaPadXDir = 'pre';
            end
            if imgSomaPadY < 0
                imgSomaPadYDir = 'post';
            end

            if frameMin < 1; frameMin = 1; end
            if frameMax > nFrames; frameMax = nFrames; end
            if xMin < 1; xMin = 1; end
            if xMax > imageWidth; xMax = imageWidth; end
            if yMin < 1; yMin = 1; end
            if yMin > imageHeight; yMin = imageHeight; end
            
            stackCellNFrames = frameMax - frameMin;
    
            % Get first image and rotate it so we know the size.
            img1 = imread(imageStackPath, 'Index', 1, 'Info', tiffInfo);
            imgDataType = class(img1);        
            img1 = padarray(img1, 1, imgSomaPadY, imgSomaPadYDir);
            img1 = padarray(img1, 2, imgSomaPadX, imgSomaPadXDir);
            img1Rot = imrotate(img1, morphData.surface_stats.angle_soma_deg, 'bilinear');
            stackCellMat = zeros(size(img1Rot, 1), size(img1Rot, 2), nFrames, imgDataType);
            stackCellMat(:, :, 1) = img1Rot;

            % Cropping to hard right now, you need to pad the image in
            % order to rotate but then that messes up the crop. Should be
            % doable but I'm lazy.
%             % Crop
%             img1 = img1(yMin:yMax, xMin:xMax);
%             img1Rot = imrotate(img1, morphData.surface_stats.angle_soma_deg, 'bilinear');
%             stackCellCropMat = zeros(size(img1Rot, 1), size(img1Rot, 2), nFrames, imgDataType);            
%             stackCellCropMat(:, :, 1) = img1Rot;
            for iFrame = 2:nFrames  
                img = imread(imageStackPath, 'Index', iFrame, 'Info', tiffInfo);
                img = padarray(img, 1, imgSomaPadY, imgSomaPadYDir);
                img = padarray(img, 2, imgSomaPadX, imgSomaPadXDir);
                imgRot = imrotate(img, morphData.surface_stats.angle_soma_deg, 'bilinear');
                imgRot = cast(imgRot, imgDataType);
                stackCellMat(:, :, iFrame) = imgRot;
%                 % Crop
%                 img = img(yMin:yMax, xMin:xMax);
%                 imgRot = imrotate(img, morphData.surface_stats.angle_soma_deg, 'bilinear');
%                 imgRot = cast(imgRot, imgDataType);
%                 stackCellCropMat(:, :, iFrame) = img;
            end
            
            stackCellMat = stackCellMat(:, :, frameMin:frameMax);
%             stackCellCropMat = stackCellCropMat(:, :, frameMin:frameMax);

%             clipLo = 5;
%             clipHi = 99.9;
%             stackCellMat = imgScaleClip(stackCellMat, clipLo, clipHi);

            stackCellMatMean = mat2gray(mean(stackCellMat, 3));
            stackCellMatMax = mat2gray(max(stackCellMat, [], 3));
 
%             stackCellMatCropMean = mat2gray(mean(stackCellCropMat, 3));
%             stackCellMatCropMax = mat2gray(max(stackCellCropMat, [], 3));
            
                                     
            imwrite(stackCellMatMean, stackCellMeanPath);
%             imwrite(stackCellMatCropMean, stackCellMeanCropPath);
            imwrite(stackCellMatMax, stackCellMaxPath);
%             imwrite(stackCellMatCropMax, stackCellMaxCropPath);
    

            
            gifDelay = 0.1;
            for idx = frameMin:stackCellNFrames  
                frame = stackCellMat(:, :, idx);
                %frameCrop = stackCellCropMat(:, :, idx);
                if idx == frameMin
                    imwrite(frame,stackCellGIFPath,'gif','LoopCount',Inf,'DelayTime', gifDelay);
%                     imwrite(frameCrop,stackCellGIFCropPath,'gif','LoopCount',Inf,'DelayTime', gifDelay);
                else
                    imwrite(frame,stackCellGIFPath,'gif','WriteMode','append','DelayTime', gifDelay);
%                     imwrite(frameCrop,stackCellGIFCropPath,'gif','WriteMode','append','DelayTime', gifDelay);
                end
            end
            
            
            clear stackCellCropMat;
        end
 
    end

end

disp('Converting metrics to table data structure ...');
metricsTable = struct2table(metricsTable)

disp('Saving metrics csv file ...');
writetable(metricsTable, dirs.metricsCSVFile, 'Delimiter', ',') 

disp('Violin plots ...');
sumPCBasic;
disp('Ephys plots ...');
sumEphysPlots;
disp('Spike waveform plots ...');
sumSpikeWaveforms;
disp('Correlations plots ...');
sumCorrelations;
disp('PCA plots ...');
sumPCPCA;
disp('Morphology plots ...');
sumAllMorphs;

disp('Done.')