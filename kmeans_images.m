function [pixel_labels,centers,ab_all,size_of_image] = kmeans_images(selectedFiles_r, n, color_bins)
    
ab_all = [];

for im=1:n
	% image = imread(selectedFiles{im});
    image = selectedFiles_r{im};
    
	image_lab = rgb2lab(image); 
	img_rgb = lab2rgb(image_lab);
	size_of_image(im,:) = size(image);
	
	ab = image_lab(:,:,2:3);
	ab = im2single(ab);
	
    ab_all = [ab_all;ab];			
	
    pl_ind=1;
	L=image_lab(:,:,1);
	a=image_lab(:,:,2);
	b=image_lab(:,:,3);
end

% pixel_labels has label of all pixels of all images
% centers has a,b color of each color_bins (coordinates)
[pixel_labels,centers] = imsegkmeans(ab_all, color_bins, 'NumAttempts', 3);

end

