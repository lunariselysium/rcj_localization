function result = estimateUndistortMetersPerPixel(ocamModel, cameraHeightM, fc, outputSize)
%ESTIMATEUNDISTORTMETERSPERPIXEL Estimate global metric scale for Scaramuzza undistort output.
%   result = estimateUndistortMetersPerPixel(ocamModel, cameraHeightM, fc, outputSize)
%   takes an already loaded ocam_model struct, mirrors the undistort plane
%   convention used by undistort.m and exportUndistortMap_v3.m, and returns
%   the global meters-per-pixel scale for the undistorted top-down approximation.

if nargin < 4 || isempty(outputSize)
    outputSize = [];
end

validateOcamModel(ocamModel);
validateattributes(cameraHeightM, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'cameraHeightM', 2);
validateattributes(fc, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, ...
    mfilename, 'fc', 3);

outputSize = resolveOutputSize(outputSize, ocamModel);

outputHeight = outputSize(1);
outputWidth = outputSize(2);
planeZ = -outputWidth / fc;
metersPerPixel = cameraHeightM / abs(planeZ);

result = struct( ...
    'meters_per_pixel', metersPerPixel, ...
    'pixels_per_meter', 1.0 / metersPerPixel, ...
    'camera_height_m', cameraHeightM, ...
    'fc', fc, ...
    'plane_z_virtual', planeZ, ...
    'output_width', outputWidth, ...
    'output_height', outputHeight, ...
    'formula', 'meters_per_pixel = cameraHeightM * fc / outputWidth', ...
    'assumptions', {{ ...
        'Camera optical axis is approximately perpendicular to the ground plane.', ...
        'The undistort virtual plane is used as a top-down ground-plane approximation.', ...
        'The returned value is a single global scale, not a full per-pixel metric model.' ...
    }} ...
);
end

function validateOcamModel(ocamModel)
if ~isstruct(ocamModel)
    error('ocamModel must be a struct loaded from ocam_model data.');
end

requiredFields = {'width', 'height'};
for idx = 1:numel(requiredFields)
    fieldName = requiredFields{idx};
    if ~isfield(ocamModel, fieldName)
        error('ocam_model is missing required field ''%s''.', fieldName);
    end
end
end

function outputSize = resolveOutputSize(outputSize, ocamModel)
if isempty(outputSize)
    outputSize = [ocamModel.height, ocamModel.width];
end

validateattributes(outputSize, {'numeric'}, {'real', 'finite', 'vector', 'numel', 2}, ...
    mfilename, 'outputSize', 4);

outputSize = double(outputSize(:).');
if any(outputSize <= 0) || any(mod(outputSize, 1) ~= 0)
    error('outputSize must be a 1x2 vector of positive integers: [height, width].');
end
end
