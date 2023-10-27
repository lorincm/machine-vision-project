load('./stereoParams.mat', 'stereoParams');

leftImage = imread('./calibration_pic/left/left_20231027_171901_0.jpg');
rightImage = imread('./calibration_pic/right/right_20231027_171901_0.jpg');

[rectifiedLeftImage, rectifiedRightImage] = rectifyStereoImages(leftImage, rightImage, stereoParams);

figure;
imshowpair(rectifiedLeftImage, rectifiedRightImage, 'montage');
title('Rectified Stereo Images (Left - Right)');
