function [spoints_all, poi_all, images_info, superPixelBoxes] = features_extraction(selectedFiles_r, n, num_of_superpixels)

    images_info = [];

    index_row_of_current_image = 1;
    for im=1:n
        % image = imread(selectedFiles{im});
        image = selectedFiles_r{im};
        image_lab = rgb2lab(image); %L*a*b
        img_rgb = lab2rgb(image_lab);
        image_gray = rgb2gray(image);
        [Labels,N] = superpixels(image,num_of_superpixels);

        region = regionprops(Labels, 'BoundingBox');
        superPixelBoxes={region.BoundingBox};      

        i=0;
        [poi_array_x, poi_array_y] = size(superPixelBoxes);
        poi_array = zeros(poi_array_x, 4, 'single');
        for superpixel = superPixelBoxes
            i = i+1;

            % Get Bounding Box coordinates 
            x = superpixel{1}(2);
            y = superpixel{1}(1);
            xwidth = superpixel{1}(4);
            yheight = superpixel{1}(3);
            xMin = ceil(x);
            xMax = xMin + xwidth - 1;
            yMin = ceil(y);
            yMax = yMin + yheight - 1;

            % Get the corresponding sub-image Is.
            Is = image_lab(x:1:xMax,y:1:yMax);

            % Extract SURF feaures
            % Determine the points of interest in the sub-image according to SURF
            % descriptor.
            spoints = detectSURFFeatures(mat2gray(Is));
            [features, valid_points] = extractFeatures(Is,spoints,'Method','SURF','SurfSize',128);
            
            % Point of Interest doesn't exist.
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
            
            spoints_all{i} = spoints;
            poi_all = poi_array;
            %%color_label_all(i,:) = I;
            % images_info = cellarray: {i} = table for each image -> each row is a superpixel with dominant color
            % images_info = table for all images -> each row is a superpixel with dominant color and poi_array (trainY and trainX)
            %%superpixel_info(i,1) = I;
            superpixel_info(i,:) = poi_array(i,:);		
        end
        
        %Gabor Features parameters
        wavelengthMin = 4/sqrt(2);
        wavelength = 2.^(0:4) * wavelengthMin; %vector
        deltaTheta = 45;
        orientation = 0:deltaTheta:(180-deltaTheta); %vector
        g = gabor(wavelength,orientation);
        gabormag = imgaborfilt(image_gray,g);

        %Extract Gabor Features for ALL superpixels. 
        K = size(gabormag,3); %K=20
        gaborFeatures = zeros(N,K);  %Initialiaze matrix for Gabor features.
        for step=1:K
           res = regionprops(Labels,gabormag(:,:,step),'MeanIntensity');
           gaborFeatures(:,step) = [res.MeanIntensity]';  
        end           
   
        images_info = poi_array;
    end
end

