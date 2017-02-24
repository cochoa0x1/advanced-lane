import numpy as np
import cv2
import json
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize


##mostly image functions

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def maybe_grayscale(img):
    '''Converts the image to gray scale if it isnt already single channel'''
    if len(img.shape) == 3 and img.shape[-1] !=1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    return gray
        
def apply_mask(img, thresh, norm=False):
    '''Applies an interval threshold to an image, returns a boolean mask'''
    if norm:
        z = np.uint8(255 * img/np.max(img))
    else:
        z = img
        
    mask = np.zeros_like(z)
    mask[ (z >= thresh[0]) & (z <= thresh[1])]=1
    return mask
    
def abs_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    '''directional sobel filter threshold'''
    gray = maybe_grayscale(img)
    if orient=='x':
        s = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient=='y':
        s = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        
    return apply_mask(s,thresh, norm=True)
    
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    '''sobel filter threshold by norm'''
    gray = maybe_grayscale(img)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    #magnitude
    m = np.sqrt(sx**2 + sy**2)
    
    return apply_mask(m,thresh, norm=True)

def angle_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''soble filter by angle'''
    gray = maybe_grayscale(img)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    theta = np.arctan2(sy,sx)
    return apply_mask(theta,thresh, norm=False)



##lane line masking

def filter_by_color(img):
    '''Use color information to find lane lines'''
    
    #1 from the first project, find yellow lines and white lines individually
    #yellow mask
    lowery = np.array([175,175,0])
    upper= np.array([255,255,150])
    yellow = cv2.inRange(img, lowery, upper)
    
    #white mask
    lowery = np.array([195,195,195])
    upper= np.array([255,255,255])
    white = cv2.inRange(img, lowery, upper)
    
    #2. convert to HLS and exploit some observations
    #HLS it
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    #split out color channels
    H, L ,S = hls[:,:,0], hls[:,:,1], hls[:,:,2] 
    
    
    #the yellow lane lines lie mostly a strong S channel and low H channel
    #while the white lines are mostly in a strong L channel
    a=apply_mask(S,(120,255))
    b=apply_mask(L,(210,255))
    c=apply_mask(H,(0,50))
    
    #mix in the first project's findings
    yellow_line = np.logical_and( np.logical_or(a,yellow), c)
    white_line = np.logical_or(b ,white)
    
    return np.logical_or(yellow_line, white_line)


def lane_line_mask(img):
    '''masks out image where lane lines might be'''
    
    #use color information
    a = filter_by_color(img)
    
    #use directional gradients
    b = abs_thresh(img,'x', sobel_kernel=3, thresh=(50,255))
    b = angle_thresh(b, sobel_kernel=3, thresh=(.75,np.pi/2))
    
    return np.maximum(a,b)
    

##curve fitting stuff    

def gauss(x,mu, sig):
    '''simple normal dist'''
    z=  np.exp( -1.0*(x-mu)**2  / (2*sig**2))
    return z / np.sum(z)

def safe_norm(q,L=640):
    '''helper function to check for a valid distribution and either normalize or return a uniform distribution'''
    if np.sum(q) ==0 or q is None:
        q=np.ones(L)
    return q/np.sum(q)
    
def observation_dist(img,y,high,low):
    '''calculates a histogram for a slice of lane line image
       the slice is runs from indexes y-low to y+high'''
    #take a slice of image and calculate a histogram
    p=np.sum(img[ max(y-low,0):y+high,:], axis=0)
    
    return safe_norm(p,img.shape[1])

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

def filter_image_by_bayes(img,x):
    '''Takes a per row probability distribution (x) and returns pixels found
       withing the interquartile range of that row's distribution'''
    
    new_im=[]
    for i in range(x.shape[0]):
        try:
            d = np.cumsum(x[i,:])
            a = np.argmax(d[d <= .25])
            b = np.argmax(d[d <= .75])
            c = np.argmax(d[d <=.5])

            a = c-max(c-a,10)
            b = c+max(b-c,10)

            new_row = np.zeros(x.shape[1])
            new_row[a:b] = img[i,a:b]
            new_im.append(new_row)
        except: #d <= ? may never happen so guard against failure by just returning the entire row
            new_im.append(img[i,:])

    return np.array(new_im)
    
    
def fit(left,right,p0,xm_per_pix=3.7/700, ym_per_pix=30.0/720, bounds=None):
    '''fits a quadratic lane line model using image of left and right lanes
    
        p0: initial guess of parameters
        bounds: interval bounds for each parameter
        
        Both left and right lane lines are fit at the same time, ie, they share paramters.
        
    '''
    def get_points(Z):
        '''get the non zero pixels in x,y coordinate'''
        y,x = np.nonzero(Z)
        return np.float64(x)*xm_per_pix , np.float64(y)*ym_per_pix
    
    xl,yl = get_points(left)
    xr,yr = get_points(right)
    
    xr+=left.shape[1]*xm_per_pix #shift the left line over a bit in the x direction
    
    def f(p):
        '''error function for fit: total MSE for both lines'''
        A,B,C,D = p
        #D=850 #fix lane width?
        pred_x_l = A*(yl**2)+B*yl+C
        pred_x_r = A*(yr**2)+B*yr+(C+D)
        return np.mean((xl-pred_x_l)**2)+ np.mean((xr-pred_x_r)**2)
    
    if bounds is None:
        bounds = [(None,None),(None,None),(100.0*xm_per_pix,400*xm_per_pix),(800.0*xm_per_pix,900.0*xm_per_pix)]
        
    thefit = minimize(f,p0,method='L-BFGS-B',bounds=bounds)
    
    #TODO check for convergence before returning results
    
    return thefit.x



#lane line finder class! 
class LaneLineFinder(object):
    '''finds lane lines in an image'''
    def __init__(self, camera_file):
        
        #constants
        self.IMAGE_X, self.IMAGE_Y = 1280,720
        
        #movement left/right per frame
        self.p_new_frame = gauss(np.arange(-100,100+1,1),mu=0, sig = 5)

        #movement left/right per increase in y
        self.p_move=gauss(np.arange(-50,50+1,1),mu=0, sig = .5)

        #scale params
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        #load the camera file
        self.mtx, self.dist = self.load_camera(camera_file)
        
        #setup the warp params
        self.M, self.M_inv = self.calc_warp_matrix()
        
        #history
        self.prior_left = None
        self.prior_right = None
        
        self.fit_params=[] #history of found lane line parameters
        self.position=[] #lane line position,radius of curve
        
    def load_camera(self, camera_file):
        '''load the camera calibration model'''
        with open(camera_file, 'r') as f:
            camera_cal = json.loads(f.read())

        mtx = np.array(camera_cal['mtx'], np.float32)
        dist = np.array(camera_cal['dist'], np.float32)
        return mtx,dist
        
    def calc_warp_matrix(self):
        '''calculates the warp matrices'''
        points = np.array([[210,self.IMAGE_Y],[595,450],[685,450],[1100,self.IMAGE_Y]], np.float32)
        dst = np.array([[210,self.IMAGE_Y],[210,0],[1100,0],[1100,self.IMAGE_Y]], np.float32)

        M = cv2.getPerspectiveTransform(points, dst)
        M_inv = cv2.getPerspectiveTransform(dst,points)
        
        return M, M_inv
    
    def lane_line_model(self,r,y):
        '''returns plotable lane lines from lane line fit parameters (r)'''
        left_fit = (r[0]*y**2+r[1]*y+r[2])/self.xm_per_pix
        right_fit = (r[0]*y**2+r[1]*y+r[2]+r[3])/self.xm_per_pix
        return left_fit, right_fit
        
    def lane_line_curve(self,r,y_eval=0):
        '''calculates road curvature'''
        return ((1 + (2*r[0]*y_eval + r[1])**2)**1.5) / np.absolute(2*r[0])
    
    def lane_line_position(self,r):
        '''calculates vehicle position relative to lane center'''
        lane_center = (r[2]+r[3]*.5)
        position =  self.IMAGE_X*self.xm_per_pix*.5 -  lane_center
        return position
        
    def draw_lines(self,r,ms=.1):
        '''creates two images for rendering purposes given fit params (r)
        
            1. A top down minimap view 
            2. A warped filled in lane line
        '''
        
        ##minimap
        
        #generate the lines
        y=np.arange(self.IMAGE_Y)
        left_fit, right_fit = self.lane_line_model(r,y*self.ym_per_pix)
        
        #calculate the curvature and position
        position=self.lane_line_position(r)
        curve = self.lane_line_curve(r)

        #create the minimap
        blank = np.zeros([self.IMAGE_Y,self.IMAGE_X]).astype(np.uint8)
        minimap = np.dstack((blank, blank, blank))
        
        y_max = self.IMAGE_Y*(1-ms)-10
        x_offset = -1*r[2]/self.xm_per_pix*ms + 2+10
        lines = [np.int32(list(zip(left_fit*ms+x_offset,y_max+y*ms))),np.int32(list(zip(right_fit*ms+x_offset,y_max+y*ms)))]
        minimap = cv2.polylines(minimap,lines,False,(0,255,0),4)
        
        #draw the car
        car_pos = (r[2]+r[3]*.5)/self.xm_per_pix*ms +x_offset
        cv2.circle(minimap,(int(car_pos),int(y_max)),4,(0,255,0),5 )
        
        minimap =np.copy(minimap[::-1,:])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        headsup = 'pos: %.2f(m) curve: %d(m)'%(position,int(curve))
        minimap=cv2.putText(minimap,headsup,(30,self.IMAGE_Y-int(y_max)+30), font, .5,(0,255,0),1,cv2.LINE_AA)
        
        ##Warped lane
        
        #create the lane overlay warped back on the road
        blank = np.zeros([self.IMAGE_Y,self.IMAGE_X]).astype(np.uint8)
        overlay = np.dstack((blank, blank, blank))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit, y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))
        
        overlay =overlay[::-1,:]
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        overlay = cv2.warpPerspective(overlay, self.M_inv, (self.IMAGE_X,self.IMAGE_Y))

        return minimap,overlay
        
    def process_img(self, img):
        '''calcuates lane line fit from an image
        
        
           pipeline:
               1. Undistort the image
               2. Warp the image to a top down view of the road
               3. Mask out the lane lines
               4. Split image into left and right halves
               5. Filter images down to remove erratic points
                   use fuzzy prior lane line position to start search
               6. Fit parameters
               7. Blend with prior fit paramteres for smoother results
               
        '''
        

        #undistort
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        #warp
        warped = cv2.warpPerspective(undist, self.M, (self.IMAGE_X,self.IMAGE_Y))
        #generate a lane line mask
        mask = lane_line_mask(warped)

        #left/right split
        L = mask[:,:int(mask.shape[1]/2)]
        R = mask[:,int(mask.shape[1]/2):]

        #flip the y values for sanity
        left = L[::-1,:]
        right = R[::-1,:]

        #remove noise
        P_left = simple_bayes_filter(left,self.p_move,prior=self.prior_left)
        P_right = simple_bayes_filter(right,self.p_move,prior=self.prior_right)
        
        #update the priors
        self.prior_left = safe_norm(np.convolve(P_left[0,:],self.p_new_frame,mode='same'),P_left.shape[1])
        self.prior_right= safe_norm(np.convolve(P_right[0,:],self.p_new_frame,mode='same'),P_right.shape[1])
        
        left= filter_image_by_bayes(left,P_left)
        right = filter_image_by_bayes(right,P_right)

        if len(self.fit_params) ==0:
            p0 = [0,0,220*self.xm_per_pix,885*self.xm_per_pix] #intial fit params
        else:
            p0 = self.fit_params[-1]
   
        r =fit(left,right,xm_per_pix=self.xm_per_pix, ym_per_pix=self.ym_per_pix, p0=p0)
    
        #simple smoothing
        blend =.5
        if len(self.fit_params) !=0:
            r = blend*r +(1-blend)*self.fit_params[-1]
        
        self.fit_params.append(r)
        
        #calculate the curvature and position
        position=self.lane_line_position(r)
        curve = self.lane_line_curve(r)
        self.position.append([position,curve])
        
        return r



