# SVM-colorization
Automatic colorization of grayscale image using SVM

For the image colorization, the following steps are performed: 
1) Conversion of RGB image to LAB color space 
2) Use of K-means clustering to reduce the number of image colors e.g. to 64.
3) Image segmentation based on superpixels using SLIC algorithm
4) SURF and Gabor feature extraction
5) Train and use of SVM for color prediction
6) Optimization using GraphCut algorithm 


In the following image the original image alongside the SVM and GraphCut output are compared using 49 (α) and 304 (β) superpixels, respectively:
![results_training_superpixels](https://user-images.githubusercontent.com/30274421/170274145-82c05612-56db-40c6-adee-9231a18a701a.png)

Additional results are shown here: 



![results](https://user-images.githubusercontent.com/30274421/170273655-0ecd7770-b2ea-4e03-9a78-073439d224f7.png)
