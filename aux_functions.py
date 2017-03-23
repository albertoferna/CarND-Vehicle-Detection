import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#########################################
# Default values obtained from test
#########################################
cspace_hog_def = 'YCrCb'   # Color space to use for hog feature extraction
cspace_color_def = 'LUV'   # Color space to use for color features extraction
spatial_def = 8            # Spatial binning, bins are spatial_def x spatial_def
hist_bins_def = 32         # Number of bins for color histogram
hist_range_def = (0, 256)  # Range of values of color
orient_def = 10            # Number of orientations to consider for binning
pix_def = 8                # Number of pixels per cell for HOG features
cell_def = 2               # Number of cells per block for HOG features
hog_def = 'ALL'            # Channels to use for HOG features
#########################################
# I should really have divided this file in several ones!

########## Feature calculation functions ###################

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
def bin_spatial(img, size=(spatial_def, spatial_def)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=hist_bins_def, bins_range=hist_range_def):
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
def extract_color_features(image, cspace=cspace_color_def,
                           spatial_size=(spatial_def, spatial_def),
                           hist_bins=hist_bins_def, hist_range=hist_range_def):
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

def extract_HOG_features(image, cspace=cspace_hog_def, orient=orient_def,
                         pix_per_cell=pix_def, cell_per_block=cell_def,
                         hog_channel=hog_def):
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
def extract_features(imgs, color_color_space=cspace_color_def,
                     hog_color_space=cspace_hog_def,
                     spatial_size=(spatial_def, spatial_def),
                     hist_bins=hist_bins_def, orient=orient_def,
                     pix_per_cell=pix_def, cell_per_block=cell_def,
                     hog_channel=hog_def, color_feat=True, hog_feat=True):
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
                                                      hist_range=hist_range_def)
            file_feat.append(image_color_feat)
        if hog_feat == True:
            image_hog_feat = extract_HOG_features(image, cspace=hog_color_space,
                                                  orient=orient, pix_per_cell=pix_per_cell,
                                                  cell_per_block=cell_per_block, hog_channel=hog_channel)
            file_feat.append(image_hog_feat)
        features.append(np.concatenate(file_feat))
    # Return list of feature vectors
    return features

def single_im_extract(img, color_color_space=cspace_color_def,
                      hog_color_space=cspace_hog_def,
                      spatial_size=(spatial_def, spatial_def),
                      hist_bins=hist_bins_def, orient=orient_def,
                      pix_per_cell=pix_def, cell_per_block=cell_def,
                      hog_channel=hog_def, color_feat=True, hog_feat=True):
    image_feat = []
    if color_feat == True:
        image_color_feat = extract_color_features(img, cspace=color_color_space, spatial_size=spatial_size,
                                                  hist_bins=hist_bins, hist_range=hist_range_def)
        image_feat.append(image_color_feat)
    if hog_feat == True:
        image_hog_feat = extract_HOG_features(img, cspace=hog_color_space, orient=orient,
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                              hog_channel=hog_channel)
        image_feat.append(image_hog_feat)
    # return image features with the required shape for a single sample
    return np.concatenate(image_feat).reshape(1, -1)
   
########## Classifier calculation functions ###################

def svm_car_classifier(cars, notcars, color_color_space=cspace_color_def,
                       hog_color_space=cspace_hog_def, spatial=spatial_def,
                       hist_bins=hist_bins_def, orient=orient_def,
                       pix_per_cell=pix_def, cell_per_block=cell_def,
                       hog_channel=hog_def, color_feat=True, hog_feat=True,
                       seed=np.random.randint(0, 100)):
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

########## Sliding window and car finding functions ###################

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

# Function to find cars in an image. Returns a list of boxes
def find_cars(img, ystart, ystop, scale, classifier, scaler, orient=orient_def,
              pix_per_cell=pix_def, cell_per_block=cell_def):
    detect_boxes = []
    draw_img = np.copy(img)
    
    img_tosearch = img[ystart:ystop,:,:]
    # I read the image with cv2.
    # YCrCb is hardcoded here as it is what I used for the classifier for hog features != colorspace for colors
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 is the sampling rate for a scale of 1
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch. Use orginal image!!! color space convertion is done during
            # feature extraction since it uses a different color space
            subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            # Get color features. Using default values hardcoded
            color_features = extract_color_features(subimg)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((color_features, hog_features)).reshape(1, -1))
            #test_prediction = classifier.predict(test_features)
            # Using decision function + threshold instead of predict
            test_prediction = classifier.decision_function(test_features)

            if test_prediction > 0.9:
            #if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                detect_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))        
    return detect_boxes

########## Heat map and filtering functions ###################
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


