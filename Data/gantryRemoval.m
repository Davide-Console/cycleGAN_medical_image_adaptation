function mask_out = gantryRemoval(img)
% GANTRYREMOVAL removes the noise caused by the gantry in the image
%
% MASK_OUT = GANTRYREMOVAL(IMAGE) removes noise from IMAGE, returning
% MASK_OUT which is the same image without noise

    mask = imbinarize(img);

    mask_out = img .* uint8(mask);
    
end