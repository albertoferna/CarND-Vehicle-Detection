import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# This function is modified from the one in the lessons to account for the framework used for images
# in this case, using cv2 to read images changes how it works as it reads in as BGR
def extract_color_features(image, cspace='BGR', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    if cspace != 'BGR':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
             feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
             feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(image)     
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    # return both vector concatenated
    return np.concatenate((spatial_features, hist_features))

def extract_HOG_features(image, cspace='BGR', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    if cspace != 'BGR':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(image)     
    
       
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, 
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Return calculated HOG features
    return hog_features

# This function extract features with separate color spaces for spatial binning/color histogram
# and hog
def extract_features(imgs, color_color_space='LUV', hog_color_space='YCrCb', spatial_size=(8, 8),
                        hist_bins=32, orient=10, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        color_feat=True, hog_feat=True):
    features = []
    # Iterate through the list of images      
    for file_name in imgs:
        # Using cv2 for consistency. it reads as BGR!!!
        image = cv2.imread(file_name)
        file_feat = []
        if color_feat == True:
            image_color_feat = extract_color_features(image, cspace=color_color_space,
                                                      spatial_size=spatial_size,
                                                      hist_bins=hist_bins,
                                                      hist_range=(0, 256))
            file_feat.append(image_color_feat)
        if hog_feat == True:
            image_hog_feat = extract_HOG_features(image, cspace=hog_color_space,
                                                  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block, hog_channel=hog_channel)
            file_feat.append(image_hog_feat)
        features.append(np.concatenate(file_feat))
    # Return list of feature vectors
    return features

def single_im_extract(img, color_color_space='LUV', hog_color_space='YCrCb', spatial_size=(8, 8),
                        hist_bins=32, orient=10, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                        color_feat=True, hog_feat=True):
    image_feat = []
    if color_feat == True:
        image_color_feat = extract_color_features(img, cspace=color_color_space, spatial_size=spatial_size,
                                                  hist_bins=hist_bins, hist_range=(0, 256))
        image_feat.append(image_color_feat)
    if hog_feat == True:
        image_hog_feat = extract_HOG_features(img, cspace=hog_color_space, orient=orient,
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                              hog_channel=hog_channel)
        image_feat.append(image_hog_feat)
    # return image features with the required shape for a single sample
    return np.concatenate(image_feat).reshape(1, -1)
   
def svm_car_classifier(cars, notcars, color_color_space='LUV', hog_color_space='YCrCb', spatial=8,
                       hist_bins=32, orient=10, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                       color_feat=True, hog_feat=True, seed=np.random.randint(0, 100)):
    ''' Takes in a list of files representing car and no car images together with parametes for
    feature straction. Returns the fitted model and its score
    '''
    print('Trainning model with binning of:',spatial, 'and',hist_bins,'histogram bins, color', color_color_space)
    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block',
          'in channel', hog_channel, 'with colorspace', hog_color_space)
    car_features = extract_features(cars, color_color_space=color_color_space, hog_color_space=hog_color_space,
                                    spatial_size=(spatial, spatial), hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                    color_feat=color_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_color_space=color_color_space, hog_color_space=hog_color_space,
                                       spatial_size=(spatial, spatial), hist_bins=hist_bins, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       color_feat=color_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print('Features:', len(scaled_X[0]))

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    # Check the score of the SVC
    score = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', round(score, 4))
    return svc, X_scaler, score


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
