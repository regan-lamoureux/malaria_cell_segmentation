# Malaria Cell Segmentation and Classification
#### This is a pipeline for segmenting and classifying UV microscope images. It uses libraries from openCV, skimage, sk-learn and follows the following pipeline:
* Segmentation
* Feature Extraction
* Classification
## Segmentation
#### Different segmentation techniques were used on these images in order to extract individual cells. Results from skimage and openCV libraries were compared to determine the best segmentation tools for this project. The same algorithm is used for both libraries, but use modules from their respective library.
#### The cell segmentation follow the following algorithm:
* Read the image and convert it to grayscale.
* Find the threshold of the image and convert to a binary image.
* Fill holes and convert to a uint8 image.
* Dilate the image and mark as sure background.
  - The dilation will increase the cell sizes slightly. We can assume the entire background of this image also lies within the entire background of the original image.
* Perform distance transform of the image and find the distance threshold (half of the distance transform max).
* Find places on the image where the distance threshold is less than the distance transform.
  - This should place markers near the center of the cell and find the places that are definitely part of the cells.
* Subtract the sure background from sure foreground.
  - This will give an unknown region between the background and foreground that must then be determined whether it is part of a cell or part of the background.
* Run connected component labeling over the sure foreground image to obtain the labeled image.
* Smooth the original grayscale image and use Sobel transform to find the edge map of the image. 
* Finally, perform watershed segmentation on the edge map image, using the labeled image as markers.
### Results:
#### Both performed similarly, correctly segmenting about 81% of the cells. However, the library from Open CV was able to better segment the complete cell and find the true border of the cell as shown by the images below:
