
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test3.jpg "Road"
[image3]: ./output_images/test3_undistort_output.png "Undistorted Road"
[image4]: ./output_images/test3_binary_combo_example.png "Binary Example"
[image5]: ./output_images/straight_lines2_warped_binary.jpg "Warp Example"
[image6]: ./output_images/window_search.png "Fit Visual Window Search"
[image7]: ./output_images/search_around_poly.png "Fit Visual Search Around Poly"
[image8]: ./output_images/output.jpg "Output"
[video1]: ./output_videos/project_video_applied_lane_finding.mp4 "Video"

---


### Camera Calibration

#### Computing of camera matrix and distortion coefficients:

The code for this step is contained in the second code cell of the IPython notebook located in "./udacity_advancedlanelines/P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function in the third code cell of the IPython notebook and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I used again the `cv2.undistort()` function in my processing pipline function called `process_image(img)` located in the 8th code cell with the image as first input for the function, after it the `mtx`, `dist` and `mtx` values to generate the distortion-corrected image.

Here is the result: 

![alt text][image3]

#### 2. Using color transforms, gradients and other methods to create a thresholded binary image.  

I used a combination of color and gradient thresholds to generate a binary image (5th code cell in IPython notebook).  Here's an example of my output for this step. 

![alt text][image4]

In detail I combined the binaries of thresholding `grayscaled` image with `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`, thresholding the `R` channel and the `B` channel. Applying `hls` with `cv2.cvtColor(img, cv2.COLOR_RGB2HLS)` and thresholding the `S` channel and the `H` channel. Also I applied the `sobel` operator with `cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)`. In code cell 5 you can see the detailed threshold values for each one.

Moreover I used the functions `region_of_interest()` and `inner_region_of_interest()` (code cell 4, used in processing pipeline) to filter out the region where the lane lines could be and blacked everything else in the binary image out.


#### 3. Performing a perspective transformation

The code for my perspective transform includes a function called `warper()`, which appears in lines 64 through 85 in the 5th code cell of the IPython notebook. The `warper()` function takes as inputs an image (`img`), as well as `img_size` and `imshape` variable.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
            [[687,450],
            [1120,imshape[0]],
            [195,imshape[0]],
            [592,450]])
            
offset = 350 # offset for dst points

dst = np.float32([[img.shape[1]-offset, 0], 
                 [img.shape[1]-offset, imshape[0]], 
                 [offset, imshape[0]], 
                 [offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 687, 450      | 930, 0        | 
| 195, 720      | 930, 720      |
| 1120, 720     | 350, 720      |
| 592, 450      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. For the perspective transformation I used the `cv2.getPerspectiveTransform()` function to generate the transformation matrix and `cv2.warpPerspective()` to warp the image to a top-down view.

Here is the warped result.

![alt text][image5]

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial

In the function `lane_line_finding` in the 7th code cell I applied a window search in the beginning and after that searching for lanes in the area of previous detected lane. It resets back to windows search when lanes are out of search area.

If searcharea lane finding is valid, the lane values will be saved in `leftlane` and `rightlane` object-values. 

To check if the lane line positions are valid, different horizontal distances from left to right lane line are measured.

The lane line positions of every frame are saved in a list if the horizontal distances sanity check (difference not greater then 60) is valid. 

With `np.average()` the lane positions are smoothed over the last 3 correctly detected lane line images. 

In detail the `window_search()` function is working like this:
In `fit_polynomial()` lane pixel are found with taking a histogram of the bootom half of the image  in the function `find_lane_pixels()` and then finding the peak of the left and right halves of the histogram. These are the starting points for left and right lane lines. After choosing the number of sliding windows and the width, also the height of the windows are calculated according to the image shape. Then all nonzero pixels in the image are found and identified which of them are inside the windows. The window positions are recentered according to the found pixels. The generated left and right laneline positions are used to fit the polynom with `np.polyfit()` to fit a second order polynomial.

The `search_around_poly()` function to search with position information from prior image defines a search area around the polynom with a specific margin and takes all nonzero image pixels in this area in consideration.

![alt text][image6]
![alt text][image7]

#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the lane curvature I implemented a function called `measure_curvature_pixels()` which can be found in the 6th code cell of the IPython notebook which is used in the processing pipeline function. In it the calculation with corresponding formula of R_curve (radius of curvature) is done with respect to real world size the conversions in x and y from pixels space to meters is integrated, therefore `laneimage.ym_per_pix` is integrated. 

Also a function `measure_vehicle_position()` is beeing called in the pipeline which can be found in the 6th code cell. There you calculate the lanecenter according to the two x positions at the bottom of the image of left and right lane and then substract it with the horizontal image center position. It is multiplied by the variable `laneimage.xm_per_pix` to get the value from pixels space to meters. 

#### 6. Providing an example image of my result plotted back down onto the road.

I implemented this step in the `process_image()` function from line 55 to 72 and used the functions `cv2.fillPoly()` and `cv2.warpPerspective()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

[Here is my video of the advanced lane finding project](https://youtu.be/mwVFD27jUsM)
![alt text][video1]

---

### Discussion

I struggled with the shadows from trees, because my S-channel filter recognised these shadows as lane pixels. After I applied a region mask for inside the two lanes the result got better.

The changing from darker street to lighter street surface was also a problem in the beginning, I fixed this by combining many different color thresholdings and channels, so it worked in both conditions. 

Also the implementation of a horziontal distance checking was a big help to filter out lanes that are not correct. 

The pipeline could be further improved in regards to shaddows and light condition changes. 

If too many frames are rejected old polynomial values still are applied which doesnt fit perfectly to the new street curvature.

Therefor it would be best to further improve the color thresholding so there are no false positives because of shadows and light conditions. 
