clc 
clear 
close all; 
warning off

%Select the images for training.
folder = uigetdir;
Files = dir(folder);
files = {Files.name};isdir = [Files.isdir];
files(isdir) = [];
s = listdlg('ListString', files);
selectedFiles = files(s);

if iscell(selectedFiles) == 0 %If select only one file, then the data will not be a cell.    
    image = imread(selectedFiles);
    if image==-1                % If this returns a -1, we did not open the file successfully.
       error('File not found or permission denied');
    end                
end

n=length(selectedFiles); %Number of selected images.
color_bins = 32; %Number of colors
num_of_superpixels = 30; %Number of superpixels

% Resize every image to 250x375 dimensions
for im=1:n
    image = imread(selectedFiles{im});
    resized = imresize(image, [250 375], 'bilinear');
    selectedFiles_r{im} = resized;
end

[pixel_labels, centers, ab_all, size_of_image] = kmeans_images(selectedFiles_r, n, color_bins); %Apply k-means to get the mean color classes of all training images.

[spoints_all, poi_all, images_info, superPixelBoxes] = features_extraction(selectedFiles_r, n, num_of_superpixels); %Extract SURF and Gabor features for all images. 

trainY = find_labelsY(selectedFiles_r, n, num_of_superpixels, color_bins, pixel_labels, size_of_image); %Get the true color labels of the training images. 


for i=1:n 
        
    image = selectedFiles_r{i};
    
    %Image representation in the Lab color space.
    image_lab = rgb2lab(image); %L*a*b
    img_rgb = lab2rgb(image_lab);

    L=image_lab(:,:,1);
    a=image_lab(:,:,2);
    b=image_lab(:,:,3);
    
%     figure
%     imshow(lab2rgb(image_lab))
%     title(sprintf('Original image before colorization', color_bins))    
       
    
    % SVM Train
    trainX = images_info;
    
    t = templateLinear();
    tTree = templateTree('surrogate','on');
    tEnsemble = templateEnsemble('GentleBoost',color_bins,tTree);
 
    model = fitcecoc(trainX',trainY,'Coding','onevsall','Learners',tEnsemble,'ObservationsIn','columns');

    [label,score] = predict(model,trainX);

    save('model.mat','model')
    
    
    
     
    % load('model.mat')
    % predictedLabels = predict(model,poi_array);

    testing_image = imread('images25.jpg');
    testing_image_lab = rgb2lab(testing_image); %L*a*b
    testing_image_gray = rgb2gray(testing_image);
    
    L=testing_image_lab(:,:,1);
    a=testing_image_lab(:,:,2);
    b=testing_image_lab(:,:,3);

    [Labels1,N1] = superpixels(testing_image_lab,num_of_superpixels); 
    
    %Gabor Features parameters
	wavelengthMin = 4/sqrt(2);
	wavelength = 2.^(0:4) * wavelengthMin; %vector
    deltaTheta = 45;
	orientation = 0:deltaTheta:(180-deltaTheta); %vector
    g = gabor(wavelength,orientation);
    gabormag_test = imgaborfilt(testing_image_gray,g);    

    %Extract Gabor Features for ALL superpixels. 
	K = size(gabormag_test,3); %K=20
    gaborFeatures = zeros(N1,K);  %Initialiaze matrix for Gabor features.
    for step=1:K
       res = regionprops(Labels1,gabormag_test(:,:,step),'MeanIntensity');
       gaborFeatures(:,step) = [res.MeanIntensity]';  
    end        

    surfFeatures = zeros(N1,128); %Initialiaze matrix for SURF features.
    for superPixelIndex = 1:N1  
        
        %Create a mask to keep only the superpixel.
        [m,n] = size(testing_image_gray);
        mask_testing = zeros(m,n);
                
        for iii = 1:m
            for j = 1:n
                if Labels1(iii,j)==superPixelIndex
                    mask_testing(iii,j) = testing_image_gray(iii,j);
                end
            end
        end        
      
        [y_superpixel, x_superpixel] = find(mask_testing); %[row, col]
        
        x_min = min(min(x_superpixel));
        x_max = max(max(x_superpixel));
        y_min = min(min(y_superpixel));
        y_max = max(max(y_superpixel));
        superpixel_image = imcrop(mask_testing,[x_min y_min x_max-x_min y_max-y_min]); %[x1 y1 x2-x1 y2-y1]
        
        % Determine the points of interest in the superpixel according to SURF
        % descriptor.
        Is = superpixel_image;

        spoints = detectSURFFeatures(mat2gray(Is));
        spoints = spoints.selectStrongest(1);

        [features, valid_points] = extractFeatures(Is,spoints,'Method','SURF','SurfSize',128); %FeatureSize can be 64 or 128
        if isempty(features)
            features = zeros(1,128); %If no points of interests are found in the superpixel, create empty array.      
        end

        surfFeatures(superPixelIndex,:) = features;
        y_ = size(Is(:,1));
        x_ = size(Is(1,:));
        dominant = zeros(color_bins, 1); 
        [size_y, size_x] = size(Is);

        sample_ = (size_x*size_y)*1; %(size_x*size_y)*0.6;
        for j = 1:sample_ %REALLY CONFUSING WITH THE x and y.... careful.
            sample_pixels(superPixelIndex,j,2) = randi([1 x_(2)],1,1) + x_min-1;
            sample_pixels(superPixelIndex,j,1) = randi([1 y_(1)],1,1) + y_min-1;
            sample_pixels(superPixelIndex,j,3) = pixel_labels(sample_pixels(superPixelIndex,j,1), sample_pixels(superPixelIndex,j,2));
            dominant(sample_pixels(superPixelIndex,j,3)) = dominant(sample_pixels(superPixelIndex,j,3)) + 1;
        end
        [M, I] = max(dominant);
        color_label(1,superPixelIndex) = I;

    end
  
%     svmInput_test = horzcat(surfFeatures, gaborFeatures);
%     
%     testX = svmInput_test';
%     testY = color_label;
%     
%     [testLabels,testScores] = predict(model,svmInput_test);
% 
%     [m,n] = size(testing_image_lab(:,:,3));
%     pixel_labels_test = zeros(m,n);
%     for superPixelIndex = 1:N1                   
%         for iii = 1:m
%             for jjj = 1:n
%                 if Labels1(iii,jjj)==superPixelIndex
%                     pixel_labels_test(iii,jjj) = testLabels(superPixelIndex);
%                 end
%             end
%         end            
%     end
        
    final(:,:,1) = L;
    final(:,:,2) = a;
    final(:,:,3) = b;
    
    
    [pixel_labels_rgb,centers_rgb] = imsegkmeans(testing_image,32);
    testing_image_colorbins = label2rgb(pixel_labels_rgb,im2double(centers_rgb));
    imshow(testing_image_colorbins)
    title('Color Quantized Image')    
    
    figure
    imshow(lab2rgb(final))
    title(sprintf('Colorized image',color_bins))  
    
    %Metrics. final: Image whose quality is to be measured. 
    %training_image_colorbins: Reference image against which quality is measured.
    psnr(lab2rgb(final), im2double(testing_image_colorbins)) %Higher is better
    mse(lab2rgb(final), im2double(testing_image_colorbins)) %Lower is better
    ssim(lab2rgb(final), im2double(testing_image_colorbins)) %Higher is better [0 1]
        
end
