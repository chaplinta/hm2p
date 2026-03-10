function dists = getSurfDist(surfPts, dendPts, somaDist)
        
    % todo this should probably been in morphology_readout
    
    % The most superficial dendrite is the one closest to the surface.
    [k, dist_super] = dsearchn(surfPts, dendPts);
    dist_super = min(dist_super);

    % The deepest dendrite is the one furthest from it's closest surface
    % point.
    [k, dist_surface] = dsearchn(surfPts, dendPts);
    dist_deep = max(dist_surface);

    dists = [dist_super, dist_deep];
end