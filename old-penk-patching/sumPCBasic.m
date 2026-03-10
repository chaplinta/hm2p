clear all;

dirs = loadPCDirs();

metricsTable = readtable(dirs.metricsCSVFile);

disp('Calculating stats and saving ...')
statsTable = sumTableNum(metricsTable, dirs.statsSumCSVFile, ...
                         dirs);


disp('Done.')