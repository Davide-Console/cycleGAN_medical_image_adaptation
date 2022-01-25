function dcm2png(filename)
% DCM2IMG converts uint16 .dcm file in uint8 .png file
%
% DCM2IMG(FILENAME) selects the image pointed by FILENAME and converts it
% inplace


%% Checking input
if ~exist('filename','var')
    error('You need an input name!');
end

dcmfile = dicomread(filename);
if ~strcmp(class(dcmfile), 'uint16') == 1
    error('uint16 is required');
end

%% Read DICOM data
newName = erase(filename, '.dcm');

metadata = dicominfo(filename);

%% Workflow
dcmfile(dcmfile>=4000)=0;

dcmImagei = uint8(225 * mat2gray(dcmfile));

%% Convert DICOM to png file

imwrite(dcmImagei, strcat(newName, '.png'), 'png');
delete(filename)

end


% PROVA 1:  dicomread                                                   [img nere]
% PROVA 2:  dicomread -> uint16(65535*mat2gray)                         [img simili (chiare)]*
% PROVA 3:  dicomread -> uint16(4095*mat2gray)                          [img nere]
% PROVA 4:  dicomread -> uint16(65535*mat2gray) -> rescale              [img simili (chiare)]
% PROVA 5:  dicomread -> rescale -> uint16(65535*mat2gray)              [img simili (scure)]*
% PROVA 6:  dicomread -> rescale -> uint16(65535*mat2gray([0 65535]))   [img nere]
% PROVA 7:  dicomread -> uint16(65535*mat2gray) -> applyVOILUT          [img saturate (corpo non distinguibile)]
% PROVA 8:  dicomread -> applyVOILUT -> uint16(65535*mat2gray)          [img saturate (corpo distinguibile)]*