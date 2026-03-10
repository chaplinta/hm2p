function [mat, colNames] = getMatrixFromTable(dataTable, prefixes, excludeCols)

    colNames = {};
    mat = [];
    colTypes = varfun(@class, dataTable, 'OutputFormat', 'cell');
    for iCol=1:width(dataTable)

        colName = string(dataTable.Properties.VariableNames{iCol});
        colData = dataTable.(iCol);
        includeCol = false;
        for iPrefix=1:numel(prefixes)
            includeCol = includeCol || startsWith(colName, prefixes{iPrefix});
            if includeCol
                break;
            end
        end
        for iExclude=1:numel(excludeCols)
            includeCol = includeCol && ~startsWith(colName, excludeCols{iExclude});
            if ~includeCol
                break;
            end
        end

        if strcmp('double', string(colTypes(iCol))) && includeCol

            mat = [mat colData];

            colNames{end+1} = convertStringsToChars(colName);
        end
    end
end