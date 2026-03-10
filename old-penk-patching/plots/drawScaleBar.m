function drawScaleBar(scale_x, scale_y, pos_x, pos_y)
    
    hold on;
    x1 = pos_x;
    x2 = x1 + scale_x;
    p1 = plot([x1 x2], ...
              [pos_y pos_y], ...
              '-', 'Color', 'k', 'LineWidth',1.5);
     
    y1 = pos_y;
    y2 = y1 + scale_y;
    
    p2 = plot([pos_x pos_x], [y1 y2], ...
         '-', 'Color', 'k', 'LineWidth', 1.5); 

    hold off;

end