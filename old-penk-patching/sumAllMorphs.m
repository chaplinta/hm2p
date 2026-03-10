clear all;

dirs = loadPCDirs();

disp('Plotting all morphologies.')

plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '001-morph-traces-main-penkpos.png'), true, true, false, false, 'penkpos');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '002-morph-traces-axon-penkpos.png'), true, true, false, true, 'penkpos');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '003-morph-traces-surf-penkpos.png'), true, true, true, true, 'penkpos');

plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '004-morph-traces-api-only-penkpos.png'), true, false, false, false, 'penkpos');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '005-morph-traces-bas-only-penkpos.png'), false, true, false, false, 'penkpos');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '006-morph-traces-axo-only-penkpos.png'), false, false, false, true, 'penkpos');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '007-morph-traces-soma-surf-penkpos.png'), false, false, true, false, 'penkpos');


plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '101-morph-traces-main-penkneg.png'), true, true, false, false, 'penkneg');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '102-morph-traces-axon-penkneg.png'), true, true, false, true, 'penkneg');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '103-morph-traces-surf-penkneg.png'), true, true, true, true, 'penkneg');

plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '104-morph-traces-api-only-penkneg.png'), true, false, false, false, 'penkneg');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '105-morph-traces-bas-only-penkneg.png'), false, true, false, false, 'penkneg');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '106-morph-traces-axo-only-penkneg.png'), false, false, false, true, 'penkneg');
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '107-morph-traces-soma-surf-penkneg.png'), false, false, true, false, 'penkneg');


plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '201-morph-density-main-only-penkpos.png'), true, true, false, false, 'penkpos', true);
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '202-morph-density-main-only-penkneg.png'), true, true, false, false, 'penkneg', true);

plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '203-morph-density-api-only-penkpos.png'), true, false, false, false, 'penkpos', true);
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '204-morph-density-api-only-penkneg.png'), true, false, false, false, 'penkneg', true);

plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '204-morph-traces-bas-only-penkpos.png'), false, true, false, false, 'penkpos', true);
plotMorphsCombined(dirs, fullfile(dirs.morphPopPlotDir, '205-morph-traces-bas-only-penkneg.png'), false, true, false, false, 'penkneg', true);


disp('Done.')