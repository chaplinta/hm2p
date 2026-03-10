function img = imgScaleClip(img, loPrc, hiPrc)

    if nargin < 2
        loPrc = 1;
    end
    if nargin < 3
        hiPrc = 99;
    end
    
    
    imgDataType = class(img); 

    prc = prctile(img(:), [loPrc, hiPrc]);

    loClip = prc(1);
    hiClip = prc(2);

    img(img < loClip) = loClip;
    img(img > hiClip) = hiClip;

    img = cast(rescale(img, 0, intmax(imgDataType)), imgDataType);

end