function  [active_p] = sp_parameters_pandora(input,nr)

%this will use PANDORA TOOLBOX https://github.com/cengique/pandora-matlab
%input=trace_pyr is the spike trace (trace at Rheobase is recommended)
%nr= if trace has more than one spike use second spike in the trace 
%
%Parameters are:
% %%%%[ 1]    'MinVm'        
%     [ 2]    'PeakVm'       
%     [ 3]    'InitVm'       
%     [ 4]    'InitVmBySlope'
%     [ 5]    'MaxVmSlope'   
%     [ 6]    'HalfVm'       
%     [ 7]    'Amplitude'    
%     [ 8]    'MaxAHP'       
%     [ 9]    'DAHPMag'      
%     [10]    'InitTime'     
%     [11]    'RiseTime'     
%     [12]    'FallTime'     
%     [13]    'MinTime'      
%     [14]    'BaseWidth'    
%     [15]    'HalfWidth'    
%     [16]    'FixVWidth'    
%     [17]    'Index'        
%     [18]    'Time'
%Pandora parameters

for i=1:size(input,2);
trace_curr=input(:,i);
dt = 1e-4;dy = 1e-3;
props = struct('spike_finder', 2, 'threshold', 0);
traces_analysis=trace(trace_curr,dt, dy, 'Analysis', props);
alltrace_info = getProfileAllSpikes(traces_analysis);
parameters=[];
parameters=alltrace_info.spikes_db.data; 
try
active_p(:,i)=parameters(nr,:);
catch
active_p(:,i)=parameters(1,:);
end;
end
end
