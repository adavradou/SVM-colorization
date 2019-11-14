function [trainY] = find_labelsY(selectedFiles_r, n, num_of_superpixels, color_bins, pixel_labels, size_of_image)

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
            
        end
        
        % Find dominant color per superpixel
        x_ = size(Is(:,1));
        y_ = size(Is(1,:));
        dominant = zeros(color_bins, 1); 
        [size_x, size_y] = size(Is);
        sample_ = (size_x*size_y)*0.6;
        for j = 1:sample_
            sample_pixels(j,1) = randi([1 x_(1)],1,1) + xMin-1;
            sample_pixels(j,2) = randi([1 y_(2)],1,1) + yMin-1;
            sample_pixels(j,3) = pixel_labels(sample_pixels(j,1), sample_pixels(j,2));
            dominant(sample_pixels(j,3)) = dominant(sample_pixels(j,3)) + 1;
        end

        index_row_of_current_image = size_of_image(im,1) + index_row_of_current_image;
        [M, I] = max(dominant);

        color_label_of_superpixel(1,i) = I;        
        
    end
    
    trainY = color_label_of_superpixel;
end
   
