clc 
clear 
close all; 

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
color_bins = 32; %Number of colors.
num_of_superpixels = 30; %Number of superpixels.

% Resize every image to 250x375 dimensions.
for im=1:n
    image = imread(selectedFiles{im});
    resized = imresize(image, [250 375], 'bilinear');
    selectedFiles_r{im} = resized;
end

%Apply k-means to get the mean color classes of all training images.
[pixel_labels, centers, ab_all, size_of_image] = kmeans_images(selectedFiles_r, n, color_bins); 
%Save centers to .txt file, in order to be used in testing.
writematrix(centers,'centers.txt','Delimiter','tab');

%Extract SURF and Gabor features for all images. 
trainX = features_extraction(selectedFiles_r, n, num_of_superpixels); 

%Get the true color labels of the training images. 
trainY = find_labelsY(selectedFiles_r, n, num_of_superpixels, color_bins, pixel_labels, size_of_image); 


%Train the SVM.
t = templateLinear();
tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',color_bins,tTree);

svm_model = fitcecoc(trainX',trainY,'Coding','onevsall','Learners',tEnsemble,'ObservationsIn','columns');
[label,score] = predict(svm_model,trainX);
disp('Training of SVM completed successfully! ')
disp([num2str(n), ' images were used for training.']);

save('svm_model.mat','svm_model')
disp('SVM model saved.')


