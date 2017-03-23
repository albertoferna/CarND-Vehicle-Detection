#Writeup Vehicle Detection Project

The work for this project is divided into several notebooks. For this reason, this writeup serves more as a guide than as a container of solutions.

##Histogram of Oriented Gradients (HOG)

The work done to obtain an histogram of oriented gradients is defined in the notebook [Vehicle Detection.ipnb](Vehicle%20Detection.ipynb).
That notebook shows the first steps into exploring alternative parameters for the feature extraction.
The functions used are defined in [aux_functions.py](aux_functions.py). Some of those functions are based of what was done during the lessons.
Some have been modified and also some functions were added. They are used throughout the notebooks.

### Parameters

After some trying of parameters I decided to to a more exhaustive. I wanted to use both hog features and color features.
With that objective, I prepared two notebooks.

One notebook deals with finding the optimum parameters for the HOG classifiers. That notebook is called [Optimize HOG Detector](Optimize%20HOG%20Detector.ipynb).
As I mention in the [vehicle detection](Vehicle%20Detection.ipynb) notebook, feature extraction for this classifier is quite costly.
For that reason I settle in assuming each parameter as independent as explained in the notebook.


The other one is called [Optimize Color Detector](Optimize%20Color%20Detector.ipynb) and it deals with selecting optimum parameters for color features.
In that notebook, I do a systematic search of color spaces, spatial grid values, and histogram bin values.
The search grid is defined in cell 6. From there, I use a function svm_car_classifier that builds a classifier and returns it together with it score.
At the bottom of the notebook I plotted the results to have a more intuitive feeling of what worked.
The final classifier is done with a spatial binning of 8, a histogram binning of 32 and LUV color space.

### Linear SVM Classifier

I trained a linear SVM in the [Vehicle Detection notebook](Vehicle%20Detection.ipynb).
It is explained in the notebook right before the code is executed in cell 8. For that training and later use, I set default values in aux_functions.py to improve readability.


##Sliding Window Search

For the windows search I started by just sliding a fixed size window across the image, but only in the region of interest.
As a first try, I set a size of 96x96 pixels for the window and an overlap of 50%. That did not seem to work well so I ended up going up to 75% overlap.
More overlap mean more computation and for that reason I tried to keep it as low as it seemed to work.
That is done in cell 11 of [Vehicle Detection notebook](Vehicle%20Detection.ipynb).
With that list of windows, I applied the classifier to each one. Results with boxes around positive are shown in cell 13 of the same notebook.

Trying to improve the result, I tested the same process with a 64x64 window. This time, knowing that it worked, I used a more complete function find_cars from [aux_functions](aux_functions.py).
Results drawn on test images are presented in cell 14.

From those results, it seemed clear that the classifier would need to be used at different scales.
For that, I used the heat map approach. In cell 15, it is applied to an image test with scales of 1, 1.25, 1.5, and 2.
Those scales correspond to windows of size 64x64, 80x80, 96x96 and 128x128.

## Video Implementation

Here's a [link to my video result](./project_video_processed.mp4)

Processing the video is done in the [Video Processing notebook](Video%20Processing.ipynb).
For the video I reduced the number of scales to try to make it run faster. In the end, I settled for 3 scales: 1, 1.5, and 2.

I also use some filtering. In cell 4 I define a variable called nframe_filter. That variable indicates the number of frames that are going to be considered in the detection of the cars.
That combined with a threshold in the heat map helped make the pipeline more stable.
As I mention in the notebook, one measure that proved very effective 
was using the decision function of the classifier instead of the prediction. That is done again in aux_functions.py


###Discussion

As I mentioned before, speed of the system seem to be slow.

I think the filtering use is very basic and not extremely helpful. Since we know that we are tracking cars the system can be made much more intelligent.
For example, we know that a car detected from one frame to the next would have some physical behaviour.
I mean, it has to have a limit in its acceleration and speed for example. It can not appear out of nowhere, etc.
That can increase computation quite a bit, but it would make the system much more robust.
Even considering what I implemented, it would seem logical to weight older frames lower that new ones.
