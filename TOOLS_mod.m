classdef TOOLS_mod
    
    methods(Static)
        function [pixel_labels,centers,ab_all,size_of_image,higher_labels] = kmeans_images(selectedFiles_r, n, color_bins)
    
        ab_all = [];
        pixel_labels={};
        centers=[];
        
        
        
        for im=1:n
            image = selectedFiles_r{im};
                    
            image_lab = rgb2lab(image); 
            
            img_rgb = lab2rgb(image_lab);
            size_of_image(im,:) = size(image);

            ab = image_lab(:,:,2:3);
            ab = im2single(ab);
            
            % pixel_labels has label of all pixels of all images
            % centers has a,b color of each color_bins (coordinates)
            [image_n_pixel_labels,image_n_centers] = imsegkmeans(ab, color_bins, 'NumAttempts', 1);
            if im == 1
               centers=image_n_centers;
            else     
               centers=cat(1,centers,image_n_centers);
            end
            pixel_labels{im}=image_n_pixel_labels;
        end
        options = statset('UseParallel',1);
        [higher_labels, centers]=kmeans(centers,color_bins,'Options',options);

        end
        
        function [images_info, Labels, i, labels_array, gaborFeatures] = features_extraction(selectedFiles_r, n, num_of_superpixels, color_bins, pixel_labels,higher_labels,gaussian)

            images_info = [];
            labels_array = [];
            i=1;
            I=0;
            index_row_of_current_image = 1;
            initial_sp_count=0;
            surf_pois_desired_per_superpixel=10;

            for im=1:n
                image = selectedFiles_r{im};
                image_lab = rgb2lab(image); %L*a*b
                img_rgb = lab2rgb(image_lab);
                image_gray = rgb2gray(image);
 
                [Labels,N] = superpixels(image,num_of_superpixels);
                
                figure
                BW = boundarymask(Labels);
                imshow(imoverlay(image_gray,BW,'cyan'));  
                title(sprintf('Image %d superpixels: %d',im,num_of_superpixels));

                region = regionprops(Labels, 'BoundingBox');
                superPixelBoxes={region.BoundingBox};      

                i=1;
                [poi_array_x, poi_array_y] = size(superPixelBoxes);
                poi_array = zeros(poi_array_x, 4, 'single');
                image_labels = zeros(1, poi_array_x*surf_pois_desired_per_superpixel, 'single');

                
                %Gabor Features parameters
                imageSize = size(image);
                numRows = imageSize(1);
                numCols = imageSize(2);
                
                %Gabor Features parameters
                wavelengthMin = 4/sqrt(2);
                wavelengthMax = hypot(numRows,numCols);
                nn = floor(log2(wavelengthMax/wavelengthMin));
                wavelength = 2.^(0:(nn-2)) * wavelengthMin;
                deltaTheta = 45;
                orientation = 0:deltaTheta:(180-deltaTheta);
                g = gabor(wavelength,orientation);                
                gabormag = imgaborfilt(image_gray,g);
                
                %gaussian filtering
                if gaussian
                    for index = 1:length(g)
                        sigma = 0.5*g(index).Wavelength;
                        K = 3;
                        gabormag(:,:,index) = imgaussfilt(gabormag(:,:,index),K*sigma); 
                    end
                end
                
                %Extract Gabor Features for ALL superpixels. 
                K = size(gabormag,3); %K=20
                gaborFeatures = zeros(N,K);  %Initialiaze matrix for Gabor features.

                for step=1:K
                   res = regionprops(Labels,gabormag(:,:,step),'MeanIntensity');
                   gaborFeatures(:,step) = [res.MeanIntensity]';  
                end   
               

                for superpixel = superPixelBoxes

                    initial_sp_count=initial_sp_count+1;
                    
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
                    
                    %%%labeling
                    if pixel_labels{im} ~= 0  
                        [size_x, size_y] = size(Is);
                        pxisp=0;
                        pxlabels_in_sp=[];
                        for ii = xMin:xMin+size_x-1
                            for jj = yMin:yMin+size_y-1
                                pxisp=pxisp+1;
                                pxlabels_in_sp(pxisp)= higher_labels(pixel_labels{im}(ii, jj));                                                               
                            end
                        end
                        I = mode(pxlabels_in_sp);
                        
                        color_label(1,i) = I;
                    end
                    
                    %%%build superpixel features

                    [szx, szy] = size(poi);
                    if (szx == 0 || szy == 0)
                        poi_array(i,1) = 0;
                        poi_array(i,2) = 0;
                        poi_array(i,3) = 0;
                        poi_array(i,4) = 0;
                        poi_array(i,5:24) = gaborFeatures(i,1:20);
                        image_labels(1,i) = I;
                    else
                        pois = valid_points.selectStrongest(surf_pois_desired_per_superpixel);
                        for p = 1:pois.Count
                            poi_array(i,1) =  pois(p).Scale;
                            poi_array(i,2) =  pois(p).SignOfLaplacian;
                            poi_array(i,3) =  pois(p).Orientation;
                            poi_array(i,4) =  pois(p).Metric;
                            poi_array(i,5:24) = gaborFeatures(i,1:20);
                            image_labels(1,i) = I;
                        end
                    end

                    spoints_all{i} = spoints;
                    poi_all = poi_array;
                    superpixel_info(i,:) = poi_array(i,:);	
                    i=i+1;
                end

                if im == 1
                   images_info = poi_array(1:i-1,:);
                   labels_array = image_labels(:,1:i-1);
                else 
                   images_info = cat(1,images_info,poi_array(1:i-1,:));
                   labels_array = cat(2,labels_array,image_labels(:,1:i-1));
                end

            end
        end
        
        function [image_features, Labels, i, gaborFeatures] = test_features_extraction2(selectedFiles_r, n, num_of_superpixels,gaussian)

            image_features = [];
            i=0;
            index_row_of_current_image = 1;
            surf_pois_desired_per_superpixel=3;

            image = selectedFiles_r;
            image_lab = rgb2lab(image); %L*a*b
            img_rgb = lab2rgb(image_lab);
            image_gray = rgb2gray(image);

           
            [Labels,N] = superpixels(image,num_of_superpixels);
             

            figure
            BW = boundarymask(Labels);
            imshow(imoverlay(image_gray,BW,'cyan'));  
            title(sprintf('Superpixels %d',num_of_superpixels));  
            
            region = regionprops(Labels, 'BoundingBox');
            superPixelBoxes={region.BoundingBox};

            i=0;
            [poi_array_x, poi_array_y] = size(superPixelBoxes);
            poi_array = zeros(poi_array_x*surf_pois_desired_per_superpixel, 5, 'single');
            labels_array = zeros(1, poi_array_x*surf_pois_desired_per_superpixel, 'single');

            %Gabor Features parameters
            imageSize = size(image);
            numRows = imageSize(1);
            numCols = imageSize(2);

            %Gabor Features parameters
            wavelengthMin = 4/sqrt(2);
            wavelengthMax = hypot(numRows,numCols);
            nn = floor(log2(wavelengthMax/wavelengthMin));
            wavelength = 2.^(0:(nn-2)) * wavelengthMin;
            deltaTheta = 45;
            orientation = 0:deltaTheta:(180-deltaTheta);
            g = gabor(wavelength,orientation);
            gabormag = imgaborfilt(image_gray,g);
            %size(gabormag)
            
            if gaussian
                for index = 1:length(g)
                    sigma = 0.5*g(index).Wavelength;
                    K = 3;
                    gabormag(:,:,index) = imgaussfilt(gabormag(:,:,index),K*sigma); 
                end
            end

            %Extract Gabor Features for ALL superpixels. 
            K = size(gabormag,3); %K=20
            gaborFeatures = zeros(N,K);  %Initialiaze matrix for Gabor features.


            for step=1:K
               res = regionprops(Labels,gabormag(:,:,step),'MeanIntensity');
               gaborFeatures(:,step) = [res.MeanIntensity]';  
            end   

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
                
                

                %%%build superpixel features
                [szx, szy] = size(poi);
                if (szx == 0 || szy == 0)
                    poi_array(i,1) = 0;
                    poi_array(i,2) = 0;
                    poi_array(i,3) = 0;
                    poi_array(i,4) = 0;            
                    poi_array(i,5:24) = gaborFeatures(i,1:20);
                else
                    poi_array(i,1) =  poi.Scale;
                    poi_array(i,2) =  poi.SignOfLaplacian;
                    poi_array(i,3) =  poi.Orientation;
                    poi_array(i,4) =  poi.Metric;
                    poi_array(i,5:24) = gaborFeatures(i,1:20);
                end

                spoints_all{i} = spoints;
                poi_all = poi_array;
            end
            image_features = poi_array;
        end

    end
end

