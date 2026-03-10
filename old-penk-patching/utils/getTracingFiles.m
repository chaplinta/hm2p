function [swcList, axonFile] = getTracingFiles(tracingFolder)

    somaFile = fullfile(tracingFolder, "Soma.swc");
    apicalFile = fullfile(tracingFolder, "Apical_tree.swc");
    surfaceFile = fullfile(tracingFolder, "Surface.swc");
    axonFile = fullfile(tracingFolder, "Axon.swc");

    basalFiles = dir(fullfile(tracingFolder, "Basal*.swc"));
    
    if ~isfile(somaFile)
        error("Could not find Soma.swc file at %s", somaFile);
    end
    if ~isfile(apicalFile)
        error("Could not find Apical_tree.swc file at %s", apicalFile);
    end
    if ~isfile(surfaceFile)
        error("Could not find Surface.swc file at %s", surfaceFile);
    end

    

    swcList = {convertStringsToChars(somaFile), ...
                convertStringsToChars(apicalFile)};

    for iBasalFile=1:length(basalFiles)
        swcList{end+1} = convertStringsToChars(fullfile(tracingFolder, ...
                                                basalFiles(iBasalFile).name));

        if ~isfile(apicalFile)
            error("Could not find Basal.swc file at %s", swcList{end});
        end
    end

    % It's ok if there is no surface file since it might not be
    % visible.
    if isfile(surfaceFile)
        surfaceFile = convertStringsToChars(surfaceFile);
        swcList{end+1} = surfaceFile;
    end
    % It's ok if there is no axon file since it might not be
    % visible.
    if isfile(axonFile)
        axonFile = convertStringsToChars(axonFile);
        swcList{end+1} = axonFile;
    end

    
    
 
end
