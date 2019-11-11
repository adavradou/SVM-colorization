% clc 
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

for i=1:n 
        
    image = imread(selectedFiles{i});
    
    %Image representation in the Lab color space.
    image_lab = rgb2lab(image); %L*a*b
    
    %Discretization of the color space.     

    %https://www.mathworks.com/help/images/color-based-segmentation-using-k-means-clustering.html
    ab = image_lab(:,:,2:3);
    ab = im2single(ab);
    color_bins = 32;
    %Repeat the clustering 3 times to avoid local minima
    [pixel_labels,centers] = imsegkmeans(ab,color_bins,'NumAttempts',3);
   
    pl_ind=1;
    img2=image_lab;
    
    L=image_lab(:,:,1);
    a=img2(:,:,1);
    b=img2(:,:,1);
    
    for idx = 1:numel(img2(:,:,2))
         a(idx)=centers(pixel_labels(idx),1);
         pl_ind=pl_ind+1;
    end
    for idx = 1:numel(img2(:,:,3))
         b(idx)=centers(pixel_labels(idx),2);
         pl_ind=pl_ind+1;
    end
    
        
    previous(:,:,1) = L;
    previous(:,:,2) = a;
    previous(:,:,3) = b;
    
    figure
    imshow(lab2rgb(previous))
    title(sprintf('Original image before colorization',color_bins))  
    
    
    
%--------------------------------------------------------------------------    

    image_gray = rgb2gray(image);
    
    %N is the actual number of superpixels that were computed.
    %L is a label matrix of type double. L is a labeled image: 
    %each pixel has the value corresponding to the ID of a superpixel. 
    %These labels start at 1 and are consecutive.
    [LabelsOfSuperpixels,NumOfSuperpixels] = superpixels(image_gray,30); 
    
%     %Visualize the superpixels.
%     figure
%     BW = boundarymask(LabelsOfSuperpixels);
%     imshow(imoverlay(image_gray,BW,'cyan'),'InitialMagnification',67);  
   

	%Gabor Features parameters
	wavelengthMin = 4/sqrt(2);
	wavelength = 2.^(0:4) * wavelengthMin; %vector
    deltaTheta = 45;
	orientation = 0:deltaTheta:(180-deltaTheta); %vector
    g = gabor(wavelength,orientation);
    gabormag = imgaborfilt(image_gray,g);    

    %Extract Gabor Features for ALL superpixels. 
	K = size(gabormag,3); %K=20
    gaborFeatures = zeros(NumOfSuperpixels,K);  %Initialiaze matrix for Gabor features.
    for step=1:K
       res = regionprops(LabelsOfSuperpixels,gabormag(:,:,step),'MeanIntensity');
       gaborFeatures(:,step) = [res.MeanIntensity]';  
    end        
    
    surfFeatures = zeros(NumOfSuperpixels,128); %Initialiaze matrix for SURF features.
    

    for superPixelIndex = 1:NumOfSuperpixels  
        
        %Create a mask to keep only the superpixel.
        [m,n] = size(image_gray);
        mask = zeros(m,n);
                
        for iii = 1:m
            for j = 1:n
                if LabelsOfSuperpixels(iii,j)==superPixelIndex
                    mask(iii,j) = image_gray(iii,j);
                end
            end
        end        
      
%         imshow(mask);           

        [y_superpixel, x_superpixel] = find(mask); %[row, col]
        
        x_min = min(min(x_superpixel));
        x_max = max(max(x_superpixel));
        y_min = min(min(y_superpixel));
        y_max = max(max(y_superpixel));
        superpixel_image = imcrop(mask,[x_min y_min x_max-x_min y_max-y_min]); %[x1 y1 x2-x1 y2-y1]
        
%         imshow(superpixel_image);     


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
        sample_ = (size_x*size_y)*0.6;
        for j = 1:sample_ %REALLY CONFUSING WITH THE x and y.... careful.
            sample_pixels(superPixelIndex,j,2) = randi([1 x_(2)],1,1) + x_min-1;
            sample_pixels(superPixelIndex,j,1) = randi([1 y_(1)],1,1) + y_min-1;
            sample_pixels(superPixelIndex,j,3) = pixel_labels(sample_pixels(superPixelIndex,j,1), sample_pixels(superPixelIndex,j,2));
            
            dominant(sample_pixels(superPixelIndex,j,3)) = dominant(sample_pixels(superPixelIndex,j,3)) + 1;
        end
        [M, I] = max(dominant);
        color_label(1,superPixelIndex) = I;

    end    
    
    svmInput = horzcat(surfFeatures, gaborFeatures);
    
    
    trainX = svmInput';
    trainY = color_label;
    

    tTree = templateTree('surrogate','on');
    tEnsemble = templateEnsemble('GentleBoost',28,tTree);

    % rng(1); % For reproducibility 
    % model = fitcecoc(trainX,trainY,'Learners',t,'ObservationsIn','columns');

    % options = statset('UseParallel',true);
    % % Mdl = fitcecoc(trainX,trainY,'Coding','onevsall','Learners',tEnsemble,...
    % %                 'Prior','uniform','NumBins',50,'Options',options);
    % 
    model = fitcecoc(trainX,trainY,'Coding','onevsall','Learners',tEnsemble,'ObservationsIn','columns');

    [label_train,score_train] = predict(model,svmInput);


    save('model.mat','model')
    % 
    % load('model.mat')
    % predictedLabels = predict(model,poi_array);


    L=image_lab(:,:,1);
    a=image_lab(:,:,2);
    b=image_lab(:,:,3);

    [Labels1,N1] = superpixels(image,30); 
    
    
    img5(:,:,1) = L;
    img5(:,:,2) = a;
    img5(:,:,3) = b;


    
%     %Visualize the superpixels.
%     figure
%     BW = boundarymask(Labels1);
%     imshow(imoverlay(image_gray,BW,'cyan'),'InitialMagnification',67);  
   

	%Gabor Features parameters
	wavelengthMin = 4/sqrt(2);
	wavelength = 2.^(0:4) * wavelengthMin; %vector
    deltaTheta = 45;
	orientation = 0:deltaTheta:(180-deltaTheta); %vector
    g = gabor(wavelength,orientation);
    gabormag = imgaborfilt(image_gray,g);    

    %Extract Gabor Features for ALL superpixels. 
	K = size(gabormag,3); %K=20
    gaborFeatures = zeros(N1,K);  %Initialiaze matrix for Gabor features.
    for step=1:K
       res = regionprops(Labels1,gabormag(:,:,step),'MeanIntensity');
       gaborFeatures(:,step) = [res.MeanIntensity]';  
    end        
    
    surfFeatures = zeros(N1,128); %Initialiaze matrix for SURF features.
    

    for superPixelIndex = 1:N1  
        
        %Create a mask to keep only the superpixel.
        [m,n] = size(image_gray);
        mask = zeros(m,n);
                
        for iii = 1:m
            for j = 1:n
                if LabelsOfSuperpixels(iii,j)==superPixelIndex
                    mask(iii,j) = image_gray(iii,j);
                end
            end
        end        
      
%         imshow(mask);           

        [y_superpixel, x_superpixel] = find(mask); %[row, col]
        
        x_min = min(min(x_superpixel));
        x_max = max(max(x_superpixel));
        y_min = min(min(y_superpixel));
        y_max = max(max(y_superpixel));
        superpixel_image = imcrop(mask,[x_min y_min x_max-x_min y_max-y_min]); %[x1 y1 x2-x1 y2-y1]
        
%         imshow(superpixel_image);     


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
        
    end
    
    svmInput_test = horzcat(surfFeatures, gaborFeatures);
    
    
    testX = svmInput_test';
    testY = color_label;
    
    
    [testLabels,testScores] = predict(model,svmInput_test);

    [m,n] = size(img5(:,:,3));
    pixel_labels_test = zeros(m,n);
    for superPixelIndex = 1:N1                   
        for iii = 1:m
            for jjj = 1:n
                if Labels1(iii,jjj)==superPixelIndex
                    pixel_labels_test(iii,jjj) = testLabels(superPixelIndex);
                end
            end
        end            
    end
    
    img2=image_lab;

    L=image_lab(:,:,1);
    a=img2(:,:,2);
    b=img2(:,:,3);
    
    for idx = 1:numel(img2(:,:,2))
         a(idx)=centers(pixel_labels_test(idx),1);
    end
    for idx = 1:numel(img2(:,:,3))
         b(idx)=centers(pixel_labels_test(idx),2);
    end    
    
    final(:,:,1) = L;
    final(:,:,2) = a;
    final(:,:,3) = b;
    
    
    figure
    imshow(lab2rgb(final))
    title(sprintf('Colorized image',color_bins))  
    
    %Metrics. final: Image whose quality is to be measured. 
    %previous: Reference image against which quality is measured.
    psnr(final, previous) %Higher is better
    mse(final, previous) %Lower is better
    ssim(final, previous) %Higher is better [0 1]
        
    
end


