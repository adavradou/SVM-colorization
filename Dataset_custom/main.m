clc 
clear 

%Choose images to read.
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

%Values that will be used for the disevetization of the color space.
min_a = 127;
min_b = 127;
max_a = -128;
max_b = -128;

for i=1:n 
        
    image = imread(selectedFiles{i});
    %image_double = double(image); %Should I convert to double?? 
    
    %Image representation in the Lab color space.
    image_lab = rgb2lab(image); %L*a*b
    
    %Discretization of the color space. 
    
    %Use of K-means is the alternative?
%     %Check https://www.mathworks.com/help/images/color-based-segmentation-using-k-means-clustering.html
%     ab = image_lab(:,:,2:3);
%     ab = im2single(ab);
%     nColors = 32;
%     % repeat the clustering 3 times to avoid local minima
%     pixel_labels = imsegkmeans(ab,nColors,'NumAttempts',3);
%     imshow(pixel_labels,[])
%     title('Image Labeled by Cluster Index');    
%     
%     %Ranges in [0, 100], and a and b in ~[-128, +127].
%     min_a_image = min(min(image_lab(:, :, 2)));
%     max_a_image = max(max(image_lab(:, :, 2)));
%     min_b_image = min(min(image_lab(:, :, 3)));
%     max_b_image = max(max(image_lab(:, :, 3)));
%     
%     %Keeps the min and max values of all images for a and b respectively.
%     if min_a_image < min_a
%         min_a = min_a_image;
%     end
%     
%     if min_b_image < min_b
%         min_b = min_b_image;
%     end
%     
%     if max_a_image > max_a
%         max_a = max_a_image;
%     end
%     
%     if max_b_image > max_b
%         max_b = max_b_image;
%     end
    

    image_gray = rgb2gray(image);
    
    %N is the actual number of superpixels that were computed.
    %L is a label matrix of type double. L is a labeled image: 
    %each pixel has the value corresponding to the ID of a superpixel. 
    %These labels start at 1 and are consecutive.
    [L,N] = superpixels(image_gray,50); 
    
    %Visualize the superpixels.
%     figure
%     BW = boundarymask(L);
%     imshow(imoverlay(image_gray,BW,'cyan'),'InitialMagnification',67);  


    %SURF Features vector initialiazation.
    surfFeatures = zeros(N, 4);
    

	%Gabor Features parameters
	wavelengthMin = 4/sqrt(2);
	wavelength = 2.^(0:4) * wavelengthMin;
    deltaTheta = 45;
	orientation = 0:deltaTheta:(180-deltaTheta);
	g = gabor(wavelength,orientation);
    gabormag = imgaborfilt(image_gray,g);        
    
%     %Postprocessing
%     %Gaussian low-pass filtering.
%     for ii = 1:length(g)
%         sigma = 0.5*g(ii).Wavelength;
%         K = 3; %Smoothing term that controls how much smoothing is applied to the Gabor magnitude responses.
%         gabormag(:,:,ii) = imgaussfilt(gabormag(:,:,ii),K*sigma); 
%     end
%     
%     %Addition of spatial location information.
%     [m,n] = size(image_gray);
%     X = 1:n;
%     Y = 1:m;
%     [X,Y] = meshgrid(X,Y);
%     featureSet = cat(3,gabormag,X);
%     featureSet = cat(3,featureSet,Y);
%     
%     %Reshape data into a matrix X of the form expected by the kmeans
%     %function.
%     numPoints = m*n;
%     X = reshape(featureSet,m*n,[]);
% 
%     %Normalize the features to be zero mean, unit variance.
%     X = bsxfun(@minus, X, mean(X));
%     X = bsxfun(@rdivide,X,std(X));    
%     
%     %Apply PCA
%     coeff = pca(X);
    


    for superPixelIndex = 1:N   
        %pixelValues = image_gray(L == superpixelindex);        
        %B = labeloverlay(image_gray,pixelValues);
        
        %Create a mask.
        [m,n] = size(image_gray);
        mask = zeros(m,n);
        
        for iii = 1:m
            for j = 1:n
                if L(iii,j)==superPixelIndex
                    mask(iii,j) = image_gray(iii,j);
                end
            end
        end
%         imshow(mask);      

        %SURF Feature extraction for each superpixel. Keep 1.
        points = detectSURFFeatures(mask);
        points = points.selectStrongest(1);
        surfFeatures(superPixelIndex,1) = points.Scale;
        surfFeatures(superPixelIndex,2) = points.Metric;
        surfFeatures(superPixelIndex,3) = points.Orientation;
        surfFeatures(superPixelIndex,4) = points.SignOfLaplacian;
           
    end     
 
  
    %Gabor Feature extraction for each superpixel.
    %The variable gaborFeatures contains one row for each superpixel, 
    %and one column for each Gabor feature.    
    K = size(gabormag,3); %K=20
    gaborFeatures = zeros(N,K);
    for step=1:K
       res = regionprops(L,gabormag(:,:,step),'MeanIntensity');
       gaborFeatures(:,step) = [res.MeanIntensity]';  
    end
    
    
end


