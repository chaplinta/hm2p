
dirs = loadPCDirs();

metricsTable = readtable(dirs.metricsCSVFile);

idxHasEphys = find(metricsTable.has_ephys_complete);
idxHasMorph = find(metricsTable.has_morph_good);
idxHasEphysMorph = find(metricsTable.has_ephys_complete & metricsTable.has_morph_good);



%% Apical length confounds
iPlot = 0;

iPlot = iPlot + 1;
xVar = "morph_api_width"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_api_depth"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_api_wd"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "depth_slice"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_api_bpoints"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_api_shlpeakcr"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_api_mbo"; 
yVar = "morph_api_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

%% Basal length confounds
iPlot = 100;

iPlot = iPlot + 1;
xVar = "morph_bas_width"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_depth"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_wd"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "depth_slice"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_bpoints"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_shlpeakcr"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_mbo"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_mpeucl"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_maxbo"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_mblen"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "morph_bas_mplen"; 
yVar = "morph_bas_len";
plotPCCorr(metricsTable, idxHasMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

%% Ephys relationships
iPlot = 200;

iPlot = iPlot + 1;
xVar = "ephys_passive_rin";
yVar = "ephys_passive_rhreo"; 
plotPCCorr(metricsTable, idxHasEphys, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "ephys_passive_incap";
yVar = "ephys_passive_rhreo"; 
plotPCCorr(metricsTable, idxHasEphys, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "ephys_passive_rin";
yVar = "ephys_passive_incap"; 
plotPCCorr(metricsTable, idxHasEphys, xVar, yVar, dirs.corrPlotDir, iPlot);

iPlot = iPlot + 1;
xVar = "ephys_passive_incap";
yVar = "ephys_active_halfWidth";
plotPCCorr(metricsTable, idxHasEphys, xVar, yVar, dirs.corrPlotDir, iPlot);

%% Ephys/morph relationships
iPlot = 300;

ephys_prop = {"ephys_passive_rhreo", ...
              "ephys_passive_rin", ...
              "ephys_passive_incap", ...
              "ephys_passive_tau", ...
              "ephys_passive_RMP", ...
              "ephys_passive_maxsp", ...
              "ephys_active_halfWidth"};

morph_prop = {"morph_api_len", ...
              "morph_bas_bpoints", ...
              "morph_api_bpoints", ...
              "morph_api_shlpeakcr", ...
              "morph_api_shlpeakcrdist", ...
              "morph_bas_len", ...
              "morph_bas_bpoints", ...
              "morph_bas_ntrees", ...
              "morph_bas_bpoints", ...
              "morph_bas_shlpeakcr", ...
              "morph_bas_shlpeakcrdist"};

for iEphys=1:length(ephys_prop)
    for iMorph=1:length(morph_prop)
        xVar = ephys_prop{iEphys};
        yVar = morph_prop{iMorph};

        iPlot = iPlot + 1;
        plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);

    end
end


% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo";
% yVar = "morph_api_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo";
% yVar = "morph_api_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo"; 
% yVar = "morph_bas_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo";
% yVar = "morph_bas_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo";
% yVar = "morph_bas_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% % Rin res vs dendrite complexity
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin"; 
% yVar = "morph_api_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin";
% yVar = "morph_api_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin";
% yVar = "morph_api_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin"; 
% yVar = "morph_bas_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin";
% yVar = "morph_bas_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin";
% yVar = "morph_bas_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% % Incap vs dendrite complexity
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap";
% yVar = "morph_api_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap";
% yVar = "morph_api_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap";
% yVar = "morph_bas_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap";
% yVar = "morph_bas_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% % Incap vs dendrite complexity
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap"; 
% yVar = "morph_api_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap"; 
% yVar = "morph_bas_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% % Tau vs dendrite complexity
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau";
% yVar = "morph_api_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau";
% yVar = "morph_api_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau";
% yVar = "morph_bas_shlpeakcr";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau";
% yVar = "morph_bas_shlpeakcrdist";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% 
% % Tau vs dendrite complexity
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau"; 
% yVar = "morph_api_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau"; 
% yVar = "morph_bas_len";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);


%% Slice depth vs passive confounds.
iPlot = 400;

for iEphys=1:length(ephys_prop)
    xVar = ephys_prop{iEphys};
    yVar = "depth_slice";

    iPlot = iPlot + 1;
    plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
end

% iPlot = iPlot + 1;
% xVar = "ephys_passive_rhreo"; 
% yVar = "depth_slice";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_rin"; 
% yVar = "depth_slice";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_incap"; 
% yVar = "depth_slice";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
% 
% iPlot = iPlot + 1;
% xVar = "ephys_passive_tau"; 
% yVar = "depth_slice";
% plotPCCorr(metricsTable, idxHasEphysMorph, xVar, yVar, dirs.corrPlotDir, iPlot);
