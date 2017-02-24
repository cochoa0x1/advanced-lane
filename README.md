## Advanced Lane Line Finder

### 1. Camera Calibration

The camera was calibrates by finding chessboard corners using `cv2.findChessboardCorners` and mapping those corners onto a calculated grid.
Those points were then fed to `cv2.calibrateCamera` to calculate the camera coefficients which were then saved to a file camera.json. The code for this is found in calibrate.ipynb

To undistort an image, the camera json file is loaded and used with `cv2.undistort`. A sample output from calibration is below:

![alt text](./output_images/calibration.png "undistored") 



### 3. Perspective Transform

I manually observed the x,y pixel coordinates of the lanes in straight lane image and calculated a perspective transform to map those points to a rectangle. Doing so transforms that section of road to one that appears as if being viewed from above. I calculate only the transformation matrix and inverse tranformation and store them in the LaneLineFinder class.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>x_dest</th>
      <th>x_org</th>
      <th>y_dest</th>
      <th>y_org</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>210.0</td>
      <td>210.0</td>
      <td>720.0</td>
      <td>720.0</td>
    </tr>
    <tr>
      <td>210.0</td>
      <td>595.0</td>
      <td>0.0</td>
      <td>450.0</td>
    </tr>
    <tr>
      <td>1100.0</td>
      <td>685.0</td>
      <td>0.0</td>
      <td>450.0</td>
    </tr>
    <tr>
      <td>1100.0</td>
      <td>1100.0</td>
      <td>720.0</td>
      <td>720.0</td>
    </tr>
  </tbody>
</table>


The org points are mapped to be transformed into the dest points
![alt text](./output_images/straight1_warp.png "warp")


```python
def calc_warp_matrix(self):
    '''calculates the warp matrices'''
    points = np.array([[210,self.IMAGE_Y],[595,450],[685,450],[1100,self.IMAGE_Y]], np.float32)
    dst = np.array([[210,self.IMAGE_Y],[210,0],[1100,0],[1100,self.IMAGE_Y]], np.float32)

    M = cv2.getPerspectiveTransform(points, dst)
    M_inv = cv2.getPerspectiveTransform(dst,points)

    return M, M_inv
```

### 3. Lane line masking

To identify lane lines a combination of color thresholds and directional gradients were used. Building on the method used in the first lane line project, white and yellow lines were identified indiviually by a color threshold.

```python
#yellow mask
lowery = np.array([175,175,0])
upper= np.array([255,255,150])
yellow = cv2.inRange(img, lowery, upper)

#white mask
lowery = np.array([195,195,195])
upper= np.array([255,255,255])
white = cv2.inRange(img, lowery, upper)
```

Additionally, by converting to an HLS colorspace, it was observed that yellow lines were mostly found at high S and low H values while white lines were found at high H values. So masks for those channels were created and ORed with the project 1 lines.

```python
a=apply_mask(S,(120,255))
b=apply_mask(L,(210,255))
c=apply_mask(H,(0,50))

#mix in the first project's findings
yellow_line = np.logical_and( np.logical_or(a,yellow), c)
white_line = np.logical_or(b ,white)

return np.logical_or(yellow_line, white_line)
```
Color alone was not reliable enough to always find the lanes, so a mix of x gradients and gradient angle thresholds were used also. The final code is in the lane_line_mask function.

![alt text](./output_images/test2_masking.png "masking")


### 4 Curve Fitting

The goal here was to fit the lane lines to a quadratic function of y. To do this a few steps were needed.

![alt text](./output_images/test2_left_process.png "line fit1")

##### 1. Split the warped and masked image into left and right images
This was fairly simple and naive, left lanes live on the left half side of the image and right live on the right half.

##### 2. Filter out unwanted points left over from overly aggresive thresholding
To cleanup the images I used simple bayseian filter that walked upwards from the bottom and constructed a per row probability distribution of where the lane lines might be.
This distribution was then used to filter out points from the source image by considering only points falling within the interquartile range and/or a fixed width window around the per row mean.

```python
def simple_bayes_filter(img,p_move,prior=None):
    '''walk vertically through a lane line image and generate a per row 
        probability distribution of where the lane line is'''
    
    ys,xs = img.shape[:2]
        
    if prior is None:
        prior = safe_norm(prior,xs)
        
    #1. sample a distribution for a set of rows,
    #2. multiply it by our prior belief of where the lane line is (the prior set of rows)
    #3. Add some uncertainty as we move vertically by convolving against p_move
    
    row_probs=[]
    for i in range(ys):

        #get our current observation distribution
        p = observation_dist(img,i,10,10)
        
        #mix with prior
        p = safe_norm(p*prior,xs)
        row_probs.append(p)
        
        #move left or right a little bit
        prior = safe_norm(np.convolve(p,p_move, mode='same'),xs)
                
    return np.array(row_probs)
```

##### 3. Calculate an appropriate loss function and pass the data to an optimizer
The model for the lines is quadratic ie Ay*2+B*y+C, and because the lines should be parallel at all y values they should have the same parameters and differ only by a translation. A fourth Paramter D was used to set the lane width and a loss function was defined as the MSE of each line summed together. This was passed to `scipy.optimize.minimize` using the L-BFGS-B solver to include optional bounds on the lane width etc.

```python
def f(p):
    '''error function for fit: total MSE for both lines'''
    A,B,C,D = p
    #D=850 #fix lane width?
    pred_x_l = A*(yl**2)+B*yl+C
    pred_x_r = A*(yr**2)+B*yr+(C+D)
    return np.mean((xl-pred_x_l)**2)+ np.mean((xr-pred_x_r)**2)
```


![alt text](./output_images/test2_fit_result.png "line fit2")



### 5. Curvature and position

Lane line curvature and vehicle position from center was calculated with the formulas below. `r` is the fit parameter vector with `r[2]` being the x position of the left lane line and `r[3]` being the width of the lane.

```python
def lane_line_curve(self,r,y_eval=0):
    '''calculates road curvature'''
    return ((1 + (2*r[0]*y_eval + r[1])**2)**1.5) / np.absolute(2*r[0])

def lane_line_position(self,r):
    '''calculates vehicle position relative to lane center'''
    lane_center = (r[2]+r[3]*.5)
    position =  self.IMAGE_X*self.xm_per_pix*.5 -  lane_center
    return position
```

### 6. Drawing!

To draw the lane lines back on the road the inverse perspective transform `M_inv` was used. I also plotted a minimap for fun in the top left. The code is in the draw_lines function.


![alt text](./output_images/test2_final_frame_result.png "final")



### Pipeline (video)

Here's a [link to my video result](./project_video_result.mp4)


<video width="960" height="540" controls>
  <source src="project_video_result.mp4">a
</video> 

---

### Discussion

The pipeline suffers from some major problems. While it attempts to track the lane lines position it currently does not do a good job of preventing wild swings in curvature etc. Additionally it is very slow. As coded producing the project_video takes about 6 minutes on a quadcore i7. Some simple heuristics might improve tolerance to false detections etc. The pipeline is also likely to fail on roads of differnet grade or S shaped curves where a higher order polynominal is needed. The reliance on color for detection is also unfortunetly critical. I doubt it would work at night or with a white or yellow car in the area of the perspective transform.