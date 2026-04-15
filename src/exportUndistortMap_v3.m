% Export a dense undistort remap table for OpenCV using Omni_Calib_Results.mat.
addpath("D:\RCJ\26\vision\fisheye_project\Scaramuzza_OCamCalib_v3.0_win\SVGA_data_correct")

% ocam_model = loadOcamModel(calibMatPath);
% Provide ocam_model in the workspace before running this script.
load('Omni_Calib_Results.mat');
ocam_model = calib_data.ocam_model;

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

projectRoot = fileparts(scriptDir);
addpath(fullfile(projectRoot, 'Scaramuzza_OCamCalib_v3.0_win'));

fc = 10;
outputSize = [];
generatePreview = true;
imagePath = fullfile(projectRoot, 'SVGA5.bmp');
outputPrefix = 'undistort_map';
% calibMatPath = fullfile(scriptDir, 'Omni_Calib_Results.mat');




if isempty(outputSize)
    outputSize = [ocam_model.height, ocam_model.width];
end

[mapX, mapY, meta] = buildUndistortMap(ocam_model, outputSize, fc);

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
mapMatPath = fullfile(scriptDir, sprintf('%s_%s.mat', outputPrefix, timestamp));
mapXmlPath = fullfile(scriptDir, sprintf('%s_%s.xml', outputPrefix, timestamp));

save(mapMatPath, 'mapX', 'mapY', 'meta', '-v7.3');
writeOpenCvXml(mapXmlPath, mapX, mapY, meta);

fprintf('Saved OpenCV remap files:\n');
fprintf('  %s\n', mapMatPath);
fprintf('  %s\n', mapXmlPath);
fprintf('mapX is the OpenCV source x/column map (zero-based).\n');
fprintf('mapY is the OpenCV source y/row map (zero-based).\n');

if generatePreview
    img = imread(imagePath);
    undistortedImg = remapImageMatlab(img, double(mapX) + 1, double(mapY) + 1);

    imagePathOut = fullfile(scriptDir, sprintf('undistorted_%s.png', timestamp));
    imwrite(undistortedImg, imagePathOut);
    fprintf('Preview image:\n');
    fprintf('  %s\n', imagePathOut);
end

% function ocam_model = loadOcamModel(calibMatPath)
% if exist(calibMatPath, 'file') ~= 2
%     error('Calibration MAT file not found in script directory:\n  %s', calibMatPath);
% end
%
% calibStruct = load(calibMatPath);
% if isfield(calibStruct, 'calib_data') && isfield(calibStruct.calib_data, 'ocam_model')
%     ocam_model = calibStruct.calib_data.ocam_model;
% elseif isfield(calibStruct, 'ocam_model')
%     ocam_model = calibStruct.ocam_model;
% else
%     error('MAT file does not contain calib_data.ocam_model:\n  %s', calibMatPath);
% end
%
% fprintf('Loaded camera model from:\n');
% fprintf('  %s\n', calibMatPath);
% end

function [mapX, mapY, meta] = buildUndistortMap(ocam_model, outputSize, fc)
if ~isfield(ocam_model, 'pol')
    maxRadius = sqrt((ocam_model.width / 2)^2 + (ocam_model.height / 2)^2);
    ocam_model.pol = findinvpoly(ocam_model.ss, maxRadius);
end

outputHeight = outputSize(1);
outputWidth = outputSize(2);
rowCenter = outputHeight / 2;
colCenter = outputWidth / 2;
planeZ = -outputWidth / fc;

[colGrid, rowGrid] = meshgrid(1:outputWidth, 1:outputHeight);
planePoints = [rowGrid(:)' - rowCenter; ...
               colGrid(:)' - colCenter; ...
               repmat(planeZ, 1, numel(rowGrid))];

sourcePoints = world2cam_fast(planePoints, ocam_model);

mapY = reshape(sourcePoints(1, :), outputHeight, outputWidth) - 1;
mapX = reshape(sourcePoints(2, :), outputHeight, outputWidth) - 1;

invalidMask = ~isfinite(mapX) | ~isfinite(mapY);
mapX(invalidMask) = -1;
mapY(invalidMask) = -1;

mapX = single(mapX);
mapY = single(mapY);

meta = struct( ...
    'fc', fc, ...
    'plane_z', planeZ, ...
    'output_height', outputHeight, ...
    'output_width', outputWidth, ...
    'source_height', ocam_model.height, ...
    'source_width', ocam_model.width);
end

function dst = remapImageMatlab(src, mapCols, mapRows)
sourceClass = class(src);
srcDouble = double(src);

if ndims(srcDouble) == 2
    dstDouble = interp2(srcDouble, mapCols, mapRows, 'linear', 0);
else
    dstDouble = zeros(size(mapRows, 1), size(mapRows, 2), size(srcDouble, 3));
    for channelIdx = 1:size(srcDouble, 3)
        dstDouble(:, :, channelIdx) = interp2(srcDouble(:, :, channelIdx), mapCols, mapRows, 'linear', 0);
    end
end

dst = cast(dstDouble, sourceClass);
end

function writeOpenCvXml(filename, mapX, mapY, meta)
fid = fopen(filename, 'w');
if fid < 0
    error('Cannot open output file: %s', filename);
end
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, '<?xml version=\"1.0\"?>\n');
fprintf(fid, '<opencv_storage>\n');
fprintf(fid, '<fc>%.12g</fc>\n', meta.fc);
fprintf(fid, '<plane_z>%.12g</plane_z>\n', meta.plane_z);
fprintf(fid, '<output_height>%d</output_height>\n', meta.output_height);
fprintf(fid, '<output_width>%d</output_width>\n', meta.output_width);
fprintf(fid, '<source_height>%d</source_height>\n', meta.source_height);
fprintf(fid, '<source_width>%d</source_width>\n', meta.source_width);

writeOpenCvMatrix(fid, 'map_x', mapX);
writeOpenCvMatrix(fid, 'map_y', mapY);
fprintf(fid, '</opencv_storage>\n');
end

function writeOpenCvMatrix(fid, name, matrix)
[rows, cols] = size(matrix);
% OpenCV FileStorage stores matrix data in row-major order.
% MATLAB stores matrices in column-major order, so transpose before
% linearizing to preserve the logical (row, col) layout on disk.
data = reshape(matrix.', 1, []);

fprintf(fid, '<%s type_id=\"opencv-matrix\">\n', name);
fprintf(fid, '  <rows>%d</rows>\n', rows);
fprintf(fid, '  <cols>%d</cols>\n', cols);
fprintf(fid, '  <dt>f</dt>\n');
fprintf(fid, '  <data>\n');

valuesPerLine = 8;
for idx = 1:numel(data)
    if mod(idx - 1, valuesPerLine) == 0
        fprintf(fid, '    ');
    end

    fprintf(fid, '%.9g', data(idx));

    if idx < numel(data)
        fprintf(fid, ' ');
    end

    if mod(idx, valuesPerLine) == 0 || idx == numel(data)
        fprintf(fid, '\n');
    end
end

fprintf(fid, '  </data>\n');
fprintf(fid, '</%s>\n', name);
end
