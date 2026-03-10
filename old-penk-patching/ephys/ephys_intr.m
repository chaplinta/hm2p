% From https://github.com/simonweiler/Electrophysiology/blob/master/ephys_intr.m

function [IV Rheobase Passive Sag Ramp] = ephys_intr(list, idx_intr, exp_folder)

 for m=1:length(idx_intr);  
    filename=[char(exp_folder) '\' list(idx_intr(m)).name];
    %load file using loadDataFile_wavesurfer
    data = loadDataFile_wavesurfer(filename);
    sr=data.header.StimulationSampleRate;
    temp=str2num(filename(end-3));

    traces=data.(['sweep_000',num2str(temp)]).analogScans(:,1);
    srF=sr/1000;
    %Filter traces
    cutoff      = 1000;     % Hz (use 500 Hz for mini event / amplitude detection and 1000Hz for max currents. Chen & Regehr 2000)
    order       = 4;        % filter order ('pole'). (use 4 pole for minis and max current. Chen & Regehr 2000)
    type        = 'Butter';
    % lowpassfilt is a custom function made by Simon that clashes with
    % pandoras. So I made a patchcells (pc) prefixed version.
    %filt_traces = lowpassfilt(traces, order, cutoff, sr, type);
    filt_traces = pc_lowpassfilt(traces, order, cutoff, sr, type);

    tracel=length(traces);
    % stimuli_type=data.header.DataFileBaseName;
    stimuli_type=filename;

    clear spike_event;
    clear spike_log;
    clear spikecount;
    clear RM_tr;
    
   
    %% 
    
    if contains(stimuli_type,'IV')==1 
           delay=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.Delay)*sr;
           delaybp=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.DelayBetweenPulses)*sr;
           pulsenr=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.PulseCount);
           pulsedur=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.PulseDuration)*sr;
           firstamp=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.FirstPulseAmplitude);
           ampch=str2num(data.header.StimulusLibrary.Stimuli.element1.Delegate.AmplitudeChangePerPulse);
           stimvec=[firstamp:ampch:(pulsenr*ampch+(firstamp-ampch))];
           %Deconcatenate traces
           startpoint=[delay/2:pulsedur+delaybp:tracel];
           endpoint=[startpoint(2):pulsedur+delaybp:tracel];
           % TC: this sometimes doesn't finds a endpoint that is out of 
           % range for the trace. See alternative bellow.
%            nStims = length(stimvec);
%            delayStart = delay/2';
%            stimDuration = pulsedur + delaybp;
%            endTime = delayStart + stimDuration * nStims;
%            startpoint = linspace(delayStart, endTime, nStims);           
%            endpoint = startpoint(2:end) + stimDuration;
           if endpoint(end) > tracel || length(stimvec) > length(endpoint)
               error(sprintf('Bad IV file %s', filename))              
           else
               
               
               for t=1:length(stimvec)
    %                t
    %                size(startpoint)
    %                size(endpoint)
    %                size(filt_traces)
    %                startpoint(t)
    %                endpoint(t)
    %                if 
    %                end
                  
               traces_deconc(:,t)=filt_traces(startpoint(t)+1:endpoint(t));
               squarepulse(:,t)=[repmat(0,delay/2,1);repmat(stimvec(t),pulsedur,1);repmat(0,delay/2,1)];
               end
               %1 : Resting membrane potential;
               for t=1:length(stimvec)
                    RM_tr(:,t)=traces_deconc(1:delay/2,t);
               end
               RMP=round(mean(RM_tr(:)),1);
               %Spike event / frequency

               for t=1:length(stimvec)
               spike_event{:,t}=spike_times(traces_deconc(:,t),0.5);
               spike_log(:,t)=~isempty(spike_event{:,t});
               spikecount(:,t)=length(spike_event{:,t});
               end
               IV.RMP=RMP;
               IV.stimvec=stimvec;
               IV.traces=traces_deconc;
               IV.spikecount=spikecount;  
               IV.header = data.header;
           end
           %% 
%Rheobase display and extraction     
        elseif contains(stimuli_type,'Rheobase')==1;
            delay=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.Delay)*sr;
            delaybp=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.DelayBetweenPulses)*sr;
            pulsenr=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.PulseCount);
            pulsedur=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.PulseDuration)*sr;
            firstamp=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.FirstPulseAmplitude);
            ampch=str2num(data.header.StimulusLibrary.Stimuli.element9.Delegate.AmplitudeChangePerPulse);
            stimvec=[firstamp:ampch:(pulsenr*ampch+(firstamp-ampch))];
            %Deconcatenate traces
            startpoint=[delay/2:pulsedur+delaybp:tracel];
            endpoint=[startpoint(2):pulsedur+delaybp:tracel];
            try
            for t=1:length(stimvec)
            traces_deconc(:,t)=filt_traces(startpoint(t)+1:endpoint(t));
            squarepulse(:,t)=[repmat(0,delay/2,1);repmat(stimvec(t),pulsedur,1);repmat(0,delay/2,1)];
            end
            catch
             for t=1:length(stimvec)-1
            traces_deconc(:,t)=filt_traces(startpoint(t)+1:endpoint(t));
            squarepulse(:,t)=[repmat(0,delay/2,1);repmat(stimvec(t),pulsedur,1);repmat(0,delay/2,1)];
            end
            end

            try
            for t=1:length(stimvec)
            spike_event{:,t}=spike_times(traces_deconc(:,t),0.5);
            spike_log(:,t)=~isempty(spike_event{:,t});
            spikecount(:,t)=length(spike_event{:,t});
            end
            catch   
             for t=1:length(stimvec)-1
            spike_event{:,t}=spike_times(traces_deconc(:,t),0.5);
            spike_log(:,t)=~isempty(spike_event{:,t});
            spikecount(:,t)=length(spike_event{:,t});
            end
            end
            spike_idx=find(spike_log==1);
            Rheob=stimvec(spike_idx(1));
            %Pandora parameters
            trace_curr=traces_deconc(:,spike_idx(1));
            dt = 1e-4;
            dy = 1e-3;
            props = struct('spike_finder', 2, 'threshold', 0);
            traces_analysis=trace(trace_curr,dt, dy, 'Analysis', props);
            alltrace_info = getProfileAllSpikes(traces_analysis);
            parameters=alltrace_info.spikes_db.data; 

            %1 : Resting membrane potential;
            for t=1:length(stimvec)
                RM_tr(:,t)=traces_deconc(1:delay/2,t);
            end
            RMP=round(mean(RM_tr(:)),1);
            
            Rheobase.parameters=parameters;
            Rheobase.rheo=Rheob;
            Rheobase.spikecount=spikecount;
            Rheobase.stimvec=stimvec;
            Rheobase.traces=traces_deconc;
            Rheobase.header = data.header;
            Rheobase.RMP = RMP;
            
                                
     %Passive display and extraction
        elseif contains(stimuli_type,'Passive')==1; 
               delay=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.Delay)*sr;
               delaybp=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.DelayBetweenPulses)*sr;
               pulsenr=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.PulseCount);
               pulsedur=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.PulseDuration)*sr;
               firstamp=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.FirstPulseAmplitude);
               ampch=str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.AmplitudeChangePerPulse);
               stimvec=[firstamp:ampch:(pulsenr*ampch+(firstamp-ampch))];
               %Deconcatenate traces
               startpoint=[delay/2:pulsedur+delaybp:tracel];
               endpoint=[startpoint(2):pulsedur+delaybp:tracel];
               traces_deconc=[];
               squarepulse=[];
               for t=1:length(stimvec)
               traces_deconc(:,t)=filt_traces(startpoint(t)+1:endpoint(t));
               squarepulse(:,t)=[repmat(0,delay/2,1);repmat(stimvec(t),pulsedur,1);repmat(0,delay/2,1)];
               end  
               Vrest_IR=mean(traces_deconc(1:delay,:));
               Vss_IR=mean(traces_deconc(2*delay:3*delay,:));
               dV_IR=Vss_IR-Vrest_IR;
               %Linear fit
               P = polyfit(stimvec',dV_IR',1);
               yfit = P(1)*stimvec+P(2);
               Passive.Rin=round(P(1)*1000,0);
              
               %tau
               tau_tr=traces_deconc(0.5*delay:delay,1:5);
               g=fittype('a-b*exp(-x/tau)');
               x=[1:length(tau_tr)];
               for j=1:5
               f0=fit(x',tau_tr(:,j),g,'Startpoint',[tau_tr(1,j) tau_tr(end,j) 0.3]);
               Passive.tau(j)=f0.tau/srF;
               end
               Passive.stimvec = stimvec;
               Passive.traces = traces_deconc;
               Passive.header = data.header;
               
        %Sag display and extraction
        elseif contains(stimuli_type,'Sag')==1;
            delay=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.Delay)*sr;
            delaybp=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.DelayBetweenPulses)*sr;
            pulsenr=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.PulseCount);
            pulsedur=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.PulseDuration)*sr;
            firstamp=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.FirstPulseAmplitude);
            ampch=str2num(data.header.StimulusLibrary.Stimuli.element5.Delegate.AmplitudeChangePerPulse);
            stimvec=ones(1,pulsenr)*firstamp;
            %Deconcatenate traces
             startpoint=[delay/2:pulsedur+delaybp:tracel];
             endpoint=[startpoint(2):pulsedur+delaybp:tracel];
             traces_deconc=[];
               squarepulse=[];
               for t=1:length(stimvec)
               traces_deconc(:,t)=filt_traces(startpoint(t)+1:endpoint(t));
               squarepulse(:,t)=[repmat(0,delay/2,1);repmat(stimvec(t),pulsedur,1);repmat(0,delay/2,1)];
               end
        Vrest_sag=mean(traces_deconc(1:startpoint(1),:));
        Vss_sag=mean(traces_deconc(2*delay:4*delay,:));
        Vmin_sag=min(traces_deconc(0.5*delay:delay,:));
        Sag.sagratio=100*((Vss_sag-Vmin_sag)./(Vrest_sag-Vmin_sag));
        Sag.traces=traces_deconc;
        Sag.header = data.header;
    %Ramp     
    elseif contains(stimuli_type,'Ramp')==1;
        delay=str2num(data.header.StimulusLibrary.Stimuli.element3.Delegate.Delay)*sr;
        duration=str2num(data.header.StimulusLibrary.Stimuli.element3.Delegate.Duration)*sr;
        endtime=data.header.StimulusLibrary.Stimuli.element3.Delegate.EndTime*sr;
        amplitude=str2num(data.header.StimulusLibrary.Stimuli.element3.Delegate.Amplitude);
        Ramp.traces=filt_traces;
        Ramp.header=data.header;
    else
            disp('Uncharted waters');
    end
                    
 end 
  if ~exist('IV')==1;
      IV=[];
  end
  
 
    if ~exist('Rheobase')==1;
      Rheobase=[];
    end
   if ~exist('Passive')==1;
      Passive=[];
   end
  
   if ~exist('Sag')==1;
      Sag=[];
   end

   if ~exist('Ramp')==1;
      Ramp=[];
  end
end