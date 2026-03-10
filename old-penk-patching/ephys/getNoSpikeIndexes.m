function noSpikeIndexes = getNoSpikeIndexes(spks)

    noSpikeIndexes = zeros(1, length(spks));
    firstSpikeIndex = find(spks > 0, 1, 'first');
    noSpikeIndexes(1:firstSpikeIndex - 1) = 1;

end