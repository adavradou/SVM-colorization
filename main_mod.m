clc 
clear 
close all; 
warning('off','all')

%Select the images for training.
folder = uigetdir;
Files = dir(folder);
files = {Files.name};isdir = [Files.isdir];
files(isdir) = [];
s = listdlg('ListString', files,'ListSize',[300,500],'Name','Training Set');
selectedFiles = files(s);

color_bins = 32; %Number of colors.
sp_ratio=1/666; %Ratio of superpixels in image
resize_scale=0.95; %Global resize rate
posterior=true;
gaussian=false;

if iscell(selectedFiles) == 0 %If select only one file, then the data will not be a cell.    
    image = imread(selectedFiles);
    if image==-1                % If this returns a -1, we did not open the file successfully.
       error('File not found or permission denied');
    end                
end

n=length(selectedFiles); %Number of selected images.


%Select the images for testing.
folder = uigetdir;
Files = dir(folder);
files = {Files.name};isdir = [Files.isdir];
files(isdir) = [];
s = listdlg('ListString', files,'ListSize',[300,500],'Name','Predict Set');
selectedPredictFiles = files(s);

if iscell(selectedPredictFiles) == 0 %If select only one file, then the data will not be a cell.    
    image = imread(selectedPredictFiles);
    if image==-1                % If this returns a -1, we did not open the file successfully.
       error('File not found or permission denied');
    end                
end

pn=length(selectedPredictFiles); %n=1.


% Resize every image to 250x375 dimensions.

resized=[];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resized=[];
% Globaly Resize every image to new scale dimensions.
if resize_scale < 1
    for im=1:n
        image = imread(selectedFiles{im});  
        [rs_x, rs_y, unecessary] = size(image);
        resized = imresize(image, resize_scale);
        selectedFiles_r{im} = resized;
    end
    for pim=1:pn
        image = imread(selectedPredictFiles{pim});  
        [rs_x, rs_y, unecessary] = size(image);
        resized = imresize(image, resize_scale);
        selectedPredictFiles_r{pim} = resized;
    end
end

        
min_x=0;
min_y=0;
%find min resolution
min_resolution=intmax('uint64');

for im=1:n
    image = selectedFiles_r{im};  
    [rs_x, rs_y, unecessary] = size(image);
    resolution=rs_x*rs_y;
    if min_resolution > resolution
        min_resolution=resolution;
        min_x=rs_x;
        min_y=rs_y;
    end
        for pim=1:pn
        image = selectedPredictFiles_r{pim};  
        [rs_x, rs_y, unecessary] = size(image);
        resolution=rs_x*rs_y;
        if min_resolution > resolution
            min_resolution=resolution;
            min_x=rs_x;
            min_y=rs_y;
        end

        end
end

% Normalizing Image resolutions

for im=1:n
    image=selectedFiles_r{im};
    [rs_x, rs_y, unecessary] = size(image);   
    new_scale=sqrt(min_resolution/(rs_x*rs_y));
    
    resized = imresize(image,  new_scale);
    selectedFiles_r{im} = resized;

        figure
        imshow(selectedFiles_r{im});  
        title(sprintf('Resized Training Image %d',im));

end
for pim=1:pn
    predict_image=selectedPredictFiles_r{pim};
    [rs_x, rs_y, unecessary] = size(predict_image);   
    new_scale=sqrt(min_resolution/(rs_x*rs_y));

    resized = imresize(predict_image,  new_scale);
    selectedPredictFiles_r{pim} = resized;

        figure
        imshow(selectedPredictFiles_r{pim});  
        title(sprintf('Resized Testing Image %d',pim));
end

num_of_superpixels=ceil(min_x*min_y*sp_ratio)

disp("------- Starting SVM Training Procedure -------")
disp("Discretizing color space...")                
%Apply k-means to get the mean color classes of all training images.
[pixel_labels, centers, ab_all, size_of_image,higher_labels] = TOOLS_mod.kmeans_images(selectedFiles_r, n, color_bins); 
%Save centers to .txt file, in order to be used in testing.
writematrix(centers,'centers.txt','Delimiter','tab');

disp("Extracting training features...")
%Extract SURF and Gabor features for all images. 
[trainX, unecessary, trainXSize, trainY, gaborFeatures]= TOOLS_mod.features_extraction(selectedFiles_r, n, num_of_superpixels, color_bins, pixel_labels, higher_labels,gaussian); 
%trainXSize
%trainXSize=size(trainX);


%Train the SVM.
t = templateLinear();
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',color_bins,tTree);

disp("Training the SVM...")
options = statset('UseParallel',true);
svm_model = fitcecoc(trainX',trainY,'FitPosterior',posterior,'Coding','onevsall','Learners',tEnsemble,'ObservationsIn','columns','Options',options);
[label,score] = predict(svm_model,trainX);
disp('Training of SVM completed successfully! ')
disp([num2str(n), ' images were used for training.']);

save('svm_model.mat','svm_model')
disp('SVM model saved.')



%Extract SURF and Gabor features for each image. 
for pim=1:pn
    disp(fprintf("------- Running prediction for image %d -------",pim))
    disp("Extracting prediction features")
    [trainX, superpixels_labels, num_of_superpixels, gaborFeatures] = TOOLS_mod.test_features_extraction2(selectedPredictFiles_r{pim}, n, num_of_superpixels,gaussian); 

    disp("Running SVM prediction")
    %Load the trained SVM model. 
    %load('svm_model.mat')
    [testLabels,testScores] = predict(svm_model,trainX);

    
    disp(fprintf("Colorizing Image",pim))
    testing_image_lab = rgb2lab(selectedPredictFiles_r{pim}); %L*a*b
    [m,n] = size(testing_image_lab(:,:,3));
    pixel_labels_test = zeros(m,n);

    for superPixelIndex = 1:num_of_superpixels                   
        for iii = 1:m
            for jjj = 1:n
                if superpixels_labels(iii,jjj)==superPixelIndex
                    pixel_labels_test(iii,jjj) = testLabels(superPixelIndex);
                end
                if testLabels(superPixelIndex) == 0
                    superPixelIndex
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

    final=[];

    final(:,:,1) = L;
    final(:,:,2) = a;
    final(:,:,3) = b;




    %Plot the original image with number of colors = colorbins.
    [pixel_labels_rgb,centers_rgb] = imsegkmeans(selectedPredictFiles_r{pim},32);
    testing_image_colorbins = label2rgb(pixel_labels_rgb,im2double(centers_rgb));


    C = unique(pixel_labels_test);
    I_gc = graph_cut(lab2rgb(final), size(C*3,1));
    I_gc = I_gc+1;
    
    L_gc=testing_image_lab(:,:,1);
    a_gc=testing_image_lab(:,:,2);
    b_gc=testing_image_lab(:,:,3);

    for idx2 = 1:numel(testing_image_lab(:,:,2))
         a_gc(idx2)=centers(I_gc(idx2),1);
    end
    for idx2 = 1:numel(testing_image_lab(:,:,3))
         b_gc(idx2)=centers(I_gc(idx2),2);
    end  
    
    final_gc=[];

    final_gc(:,:,1) = L_gc;
    final_gc(:,:,2) = a_gc;
    final_gc(:,:,3) = b_gc;


    %Plot the original image with number of colors = colorbins.
    [pixel_labels_rgb,centers_rgb] = imsegkmeans(selectedPredictFiles_r{pim},color_bins);
    testing_image_colorbins = label2rgb(pixel_labels_rgb,im2double(centers_rgb));
    figure
    imshow(testing_image_colorbins)
    title('Color Quantized Image')    

    %Plot the resulting image after colorization. 
    figure
    imshow(lab2rgb(final))
    title(sprintf('Colorized image %d -%d - %d ',color_bins,1/sp_ratio))  
    
    %Plot the resulting image after colorization. 
    figure
    imshow(lab2rgb(final_gc))
    title(sprintf('After graph-cuts',color_bins)) 

    %Evaluate the output using metrics. 
    %Metrics. final: Image whose quality is to be measured. 
    %training_image_colorbins: Reference image against which quality is measured.
    disp(fprintf("Output evaluation using metrics for image %d : "))
    try
        psnr = psnr(lab2rgb(final_gc), im2double(testing_image_colorbins)) %Higher is better
        mean = mean(mse(lab2rgb(final_gc), im2double(testing_image_colorbins))) %Lower is better
        ssim = ssim(lab2rgb(final_gc), im2double(testing_image_colorbins)) %Higher is better [0 1]    
    catch exception
       disp(fprintf("Couldn't calculate statistics for image %d, maybe this is a greyscale image",pim))
    end
    
end