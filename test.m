clc 
clear 
close all; 
warning off


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
%  (image)
%     title('Original RGB Loaded Image');
%     
    img_rgb = lab2rgb(image_lab);
%     figure
%     imshow(img_rgb,[])
%     title('Original Lab Image (after double conversion)');

    %Discretization of the colorspace -----------------------------------------    
    
    ab = image_lab(:,:,2:3);
    ab = im2single(ab);
    color_bins = 64;
    % repeat the clustering 3 times to avoid local minima
    [pixel_labels,centers] = imsegkmeans(ab,color_bins,'NumAttempts',5);
    size(centers)
    pl_ind=1;
    img2=image_lab;

    L=image_lab(:,:,1);
    a=img2(:,:,2);
    b=img2(:,:,3);
    
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
    
    image_gray = rgb2gray(image);
    
    [Labels,N] = superpixels(image,30); 

%     figure
    BW = boundarymask(Labels);
%     imshow(imoverlay(image_gray,BW,'cyan'));    
    
    region = regionprops(Labels, 'BoundingBox');
    superPixelBoxes={region.BoundingBox};          
     
    figure
    imshow(label2rgb(Labels))
    title(sprintf('L proper %d Colors',color_bins))
    
    
    i=0;
    [poi_array_x, poi_array_y] = size(superPixelBoxes);
    poi_array = zeros(poi_array_x, 4, 'single');
    for superpixel = superPixelBoxes
        i = i+1;
        % Get the corresponding random sub-image Is.  
        %Is = image(superpixelBoundaries(1):1:superpixelBoundaries(2),superpixelBoundaries(3):1:superpixelBoundaries(4));

        % Get a random (xo,yo) point in the upper-left half of the image plane.
        x = superpixel{1}(2);
        y = superpixel{1}(1);

        xwidth = superpixel{1}(4);
        yheight = superpixel{1}(3);

        xMin = ceil(x);
        xMax = xMin + xwidth - 1;
        yMin = ceil(y);
        yMax = yMin + yheight - 1;

        % Get the corresponding random sub-image Is.
        Is = image_lab(x:1:xMax,y:1:yMax);

        % Extract SURF feaures
        % Determine the points of interest in the sub-image according to SURF
        % descriptor.
        spoints = detectSURFFeatures(mat2gray(Is));
        [features, valid_points] = extractFeatures(Is,spoints,'Method','SURF','SurfSize',128);

        % Show the sub-image pinpointing the extracted SURF features.
        poi = valid_points.selectStrongest(1);
        [szx, szy] = size(poi);
        if (szx == 0 || szy == 0)
            poi_array(i,1) = 0;
            poi_array(i,2) = 0;
            poi_array(i,3) = 0;
            poi_array(i,4) = 0;
        else
            poi_array(i,1) = poi.Scale;
            poi_array(i,2) = poi.SignOfLaplacian;
            poi_array(i,3) = poi.Orientation;
            poi_array(i,4) = poi.Metric;
        end
%         figure;
%         imshow(mat2gray(Is));
%         hold on;
%         plot(poi,'showOrientation',true);
%         hold off
%         Lab_Is = Is;

%         c_ = {};
%         for k=1:10
            x_ = size(Is(:,1));
            y_ = size(Is(1,:));
            dominant = zeros(color_bins, 1); 
            [size_x, size_y] = size(Is);
            sample_ = (size_x*size_y)*0.6;
            for j = 1:sample_
                sample_pixels(i,j,1) = randi([1 x_(1)],1,1) + xMin-1;
                sample_pixels(i,j,2) = randi([1 y_(2)],1,1) + yMin-1;
                sample_pixels(i,j,3) = pixel_labels(sample_pixels(i,j,1), sample_pixels(i,j,2));
                dominant(sample_pixels(i,j,3)) = dominant(sample_pixels(i,j,3)) + 1;
            end
            [M, I] = max(dominant);
            color_label(1,i) = I;
%             c_{k} = color_label;
%         end
    end
  
    
    
 
t = templateLinear();


trainX = poi_array';
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

[label,score] = predict(model,poi_array);


save('model.mat','model')
% 
% load('model.mat')
% predictedLabels = predict(model,poi_array);


L=image_lab(:,:,1);
a=image_lab(:,:,2);
b=image_lab(:,:,3);

[Labels1,N1] = superpixels(image,30); 

for idx = 1:numel(image_lab(:,:,2))
     a(idx)=centers(pixel_labels(idx),1);
end
for idx = 1:numel(image_lab(:,:,3))
     b(idx)=centers(pixel_labels(idx),2);
end

img5(:,:,1)=L;
img5(:,:,2)=a;
img5(:,:,3)=b;



    ii=0;
    [poi_array_x, poi_array_y] = size(superPixelBoxes);
    poi_array = zeros(poi_array_x, 4, 'single');
    for superpixel = superPixelBoxes
        ii = ii+1;
        % Get the corresponding random sub-image Is.  
        %Is = image(superpixelBoundaries(1):1:superpixelBoundaries(2),superpixelBoundaries(3):1:superpixelBoundaries(4));

        % Get a random (xo,yo) point in the upper-left half of the image plane.
        x = superpixel{1}(2);
        y = superpixel{1}(1);

        xwidth = superpixel{1}(4);
        yheight = superpixel{1}(3);

        xMin = ceil(x);
        xMax = xMin + xwidth - 1;
        yMin = ceil(y);
        yMax = yMin + yheight - 1;

        % Get the corresponding random sub-image Is.
        Is = image_lab(x:1:xMax,y:1:yMax);

        % Extract SURF feaures
        % Determine the points of interest in the sub-image according to SURF
        % descriptor.
        spoints = detectSURFFeatures(mat2gray(Is));
        [features, valid_points] = extractFeatures(Is,spoints,'Method','SURF','SurfSize',128);

        % Show the sub-image pinpointing the extracted SURF features.
        poi = valid_points.selectStrongest(1);
        [szx, szy] = size(poi);
        if (szx == 0 || szy == 0)
            poi_array(ii,1) = 0;
            poi_array(ii,2) = 0;
            poi_array(ii,3) = 0;
            poi_array(ii,4) = 0;
        else
            poi_array(ii,1) = poi.Scale;
            poi_array(ii,2) = poi.SignOfLaplacian;
            poi_array(ii,3) = poi.Orientation;
            poi_array(ii,4) = poi.Metric;
        end

    end
  
    
    

    t = templateLinear();


    trainX2 = poi_array';

    [label5,score5] = predict(model,poi_array);

    [m,n] = size(img5(:,:,3));
    pixel_labels_2 = zeros(m,n);
    for superPixelIndex = 1:N   

%         pixel_labels_2 = zeros(m,n);
                
        for iii = 1:m
            for jjj = 1:n
                if Labels1(iii,jjj)==superPixelIndex
                    pixel_labels_2(iii,jjj) = label5(superPixelIndex);
                end
            end
        end            
    end
    
    img2=image_lab;

    L=image_lab(:,:,1);
    a=img2(:,:,2);
    b=img2(:,:,3);
    
    for idx = 1:numel(img2(:,:,2))
         a(idx)=centers(pixel_labels_2(idx),1);
    end
    for idx = 1:numel(img2(:,:,3))
         b(idx)=centers(pixel_labels_2(idx),2);
    end    
    
    final(:,:,1) = L;
    final(:,:,2) = a;
    final(:,:,3) = b;
    
    
    figure
    imshow(lab2rgb(final))
    title(sprintf('Colorized image',color_bins))    


end
