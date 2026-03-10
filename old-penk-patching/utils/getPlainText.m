% from https://uk.mathworks.com/matlabcentral/answers/94238-how-can-i-place-the-_-or-characters-in-a-text-command#:~:text=Accepted%20Answer,-MathWorks%20Support%20Team&text=The%20underscore%20character%20%22%20_%20%22%20is,MATLAB%20as%20a%20superscript%20command.
function s = getPlainText(s)
    % hide super/sub/backslash from Tex interpreter 
    % (e.g. when putting a windows path\file_name on a plot)
    s = regexprep( s, '[\\\^\_]','\\$0');
end