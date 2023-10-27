% Load the stereoParams object from your .mat file
load('stereoParams.mat', 'stereoParams'); % replace 'your_file_name.mat' with the actual name

% Create a simplified structure to store the required parameters
simpleStereoParams = struct();

% Extract intrinsic matrices
simpleStereoParams.CameraMatrix1 = stereoParams.CameraParameters1.IntrinsicMatrix';
simpleStereoParams.CameraMatrix2 = stereoParams.CameraParameters2.IntrinsicMatrix';

% Extract distortion coefficients
simpleStereoParams.DistCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion, stereoParams.CameraParameters1.TangentialDistortion];
simpleStereoParams.DistCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion, stereoParams.CameraParameters2.TangentialDistortion];

% Extract rotation and translation of camera 2
simpleStereoParams.RotationOfCamera2 = stereoParams.RotationOfCamera2;
simpleStereoParams.TranslationOfCamera2 = stereoParams.TranslationOfCamera2;

% Save the simplified structure to a new .mat file
save('simplifiedStereoParams.mat', '-struct', 'simpleStereoParams');
