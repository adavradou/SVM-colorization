clc 
clear 
% close all; 

%Selected image for testing.
selectedFiles = {'images23.jpg'};

n=length(selectedFiles); %n=1.
color_bins = 32; %Number of colors.
num_of_superpixels = 30; %Number of superpixels.

% Resize every image to 250x375 dimensions.
image = imread(selectedFiles{1});
resized = imresize(image, [250 375], 'bilinear');
selectedFiles_r{1} = resized;

%Extract SURF and Gabor features for all images. 
[trainX, superpixels_labels] = features_extraction(selectedFiles_r, n, num_of_superpixels); 

%Load the trained SVM model. 
load('svm_model.mat')
[testLabels,testScores] = predict(svm_model,trainX);

testing_image_lab = rgb2lab(selectedFiles_r{1}); %L*a*b

[m,n] = size(testing_image_lab(:,:,3));
pixel_labels_test = zeros(m,n);

for superPixelIndex = 1:num_of_superpixels                   
    for iii = 1:m
        for jjj = 1:n
            if superpixels_labels(iii,jjj)==superPixelIndex
                pixel_labels_test(iii,jjj) = testLabels(superPixelIndex);
            end
        end
    end            
end

L=testing_image_lab(:,:,1);
a=testing_image_lab(:,:,2);
b=testing_image_lab(:,:,3);

%Read the centers of color that were computed through kmeans in training. 
centers = dlmread('centers.txt'); 
centers = im2single(centers);


for idx = 1:numel(testing_image_lab(:,:,2))
     a(idx)=centers(pixel_labels_test(idx),1);
end
for idx = 1:numel(testing_image_lab(:,:,3))
     b(idx)=centers(pixel_labels_test(idx),2);
end    

final(:,:,1) = L;
final(:,:,2) = a;
final(:,:,3) = b;


%Plot the original image with number of colors = colorbins.
[pixel_labels_rgb,centers_rgb] = imsegkmeans(selectedFiles_r{1},32);
testing_image_colorbins = label2rgb(pixel_labels_rgb,im2double(centers_rgb));
figure
imshow(testing_image_colorbins)
title('Color Quantized Image')    

%Plot the resulting image after colorization. 
% figure
imshow(lab2rgb(final))
title(sprintf('Colorized image',color_bins))  

%Evaluate the output using metrics. 
  %Metrics. final: Image whose quality is to be measured. 
  %training_image_colorbins: Reference image against which quality is measured.
psnr(lab2rgb(final), im2double(testing_image_colorbins)) %Higher is better
mean(mse(lab2rgb(final), im2double(testing_image_colorbins))) %Lower is better
ssim(lab2rgb(final), im2double(testing_image_colorbins)) %Higher is better [0 1]
        

