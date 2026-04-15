% Simple runner for estimateUndistortMetersPerPixel.m
% Edit the parameters below, then run this script in MATLAB.

scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(scriptDir);

% Input parameters
calibMatPath = fullfile(scriptDir, 'Omni_Calib_Results.mat');
cameraHeightM = 0.80;
fc = 10;
outputSize = [600, 800];  % [height, width]

calibStruct = load(calibMatPath);
if isfield(calibStruct, 'calib_data') && isfield(calibStruct.calib_data, 'ocam_model')
    ocamModel = calibStruct.calib_data.ocam_model;
elseif isfield(calibStruct, 'ocam_model')
    ocamModel = calibStruct.ocam_model;
else
    error(['MAT file does not contain calib_data.ocam_model or ocam_model:\n  %s'], ...
        calibMatPath);
end

result = estimateUndistortMetersPerPixel( ...
    ocamModel, ...
    cameraHeightM, ...
    fc, ...
    outputSize);

fprintf('estimateUndistortMetersPerPixel result:\n');
fprintf('  meters_per_pixel : %.9f\n', result.meters_per_pixel);
fprintf('  pixels_per_meter : %.9f\n', result.pixels_per_meter);
fprintf('  camera_height_m  : %.6f\n', result.camera_height_m);
fprintf('  fc               : %.6f\n', result.fc);
fprintf('  plane_z_virtual  : %.6f\n', result.plane_z_virtual);
fprintf('  output_width     : %d\n', result.output_width);
fprintf('  output_height    : %d\n', result.output_height);
fprintf('  formula          : %s\n', result.formula);

disp('  assumptions:');
disp(result.assumptions);
disp('  full result struct:');
disp(result);
