%%
close; clear; clc;

%% Converting uint16 .dcm image to uint8 .png image
path = pwd;

contents = dir(fullfile(path, '*\*\*.dcm'));

h = waitbar(i/numel(contents), 'Conversion uint16 .dcm files to uint8 .png files in progress...');
for i = 1:numel(contents)
    waitbar(i/numel(contents), h);
    
    filename = strcat(contents(i).folder, '\', contents(i).name);
    dcm2png(filename);

end
close(h);

%% Setting all images to 256x256 && Deleting Gantry
contents = dir(fullfile(path, '*\*\*.png'));

h = waitbar(0, '256x256 resizing and gantry removal in progress...');
for i = 1:numel(contents)
    waitbar(i/numel(contents), h);

    filename = strcat(contents(i).folder, '\', contents(i).name);

    img = imread(filename);
    img = imresize(img, [256 256]);
    img = gantryRemoval(img);

    imwrite(img, filename, 'png');

end
close(h);
