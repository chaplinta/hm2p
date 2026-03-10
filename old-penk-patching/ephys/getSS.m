function VssDiff = getSS(data)

    sF=data.header.StimulationSampleRate;
    delay = str2num(data.header.StimulusLibrary.Stimuli.element6.Delegate.Delay)*sF;
    Vrest = mean(data.traces(1:delay,:));
    Vss = mean(data.traces(2*delay:3*delay,:));
    VssDiff = Vss - Vrest;

end