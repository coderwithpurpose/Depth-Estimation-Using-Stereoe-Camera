import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
# import  generate_pointcloud as get_mesh
import copy
# import pptk
import time


zed_2k_l ={'cx': 1088.61 , 'cy': 606.78, 'fx': 1400.33 , 'fy': 1400.33, 'k1': -0.170457 , 'k2': 0.0268623}
zed_2k_r ={'cx':1071.51 , 'cy': 590.985, 'fx': 11395.84 , 'fy': 1395.84, 'k1': -0.169439 , 'k2': 0.026557}
zed_2k_s = {'BaseLine': 119.966,'CV_2K':0.00854033,'RZ_2K':-0.001123,'RX_2K':-0.00720126}
zed_hd_l= {'cx': 630.805 , 'cy': 351.39, 'fx': 700.163, 'fy': 700.163 , 'k1': -0.170457 ,'k2': 0.0268623}
zed_hd_r= {'cx': 622.254 , 'cy': 343.492, 'fx': 697.921, 'fy': 697.921 , 'k1': -0.169439 ,'k2': 0.026557}
zed_hd_s = {'BaseLine': 119.966,'CV_HD':0.00854033,'RZ_HD': -0.001123,'RX_HD':-0.00720126}

def get_distance(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # pts = cv2.reprojectImageTo3D(disparity=disp2[y-1:y+1,x-1:x+1], Q=Q )
        # print('3d pts = ', pts/1000)
        Distance = (f*base*0.001)/disp2[y,x]
        # Distance= np.around(Distance*0.001,decimals=3)
        print('Distance: '+ str(Distance)+' m')

def cam_calibrate(mode):
    if mode == 'HD':
        params_l,params_r,ext_pars = zed_hd_l,zed_hd_r,zed_hd_s
    elif mode == '2K':
        params_l, params_r,ext_pars = zed_2k_l, zed_2k_r,zed_2k_s
    else:
        params_l,params_r,ext_pars = False,False,False
    return  params_l,params_r,ext_pars

def Rectify_Params(zed_s,mode):
    T = np.array([-zed_s['BaseLine'], 0, 0])
    Rz, _ = cv2.Rodrigues(np.array([0, 0, zed_s['RZ_'+mode]]))
    Ry, _ = cv2.Rodrigues(np.array([0, zed_s['CV_'+mode], 0]))
    Rx, _ = cv2.Rodrigues(np.array([zed_s['RX_'+mode], 0, 0]))
    R = np.dot(Rz, np.dot(Ry, Rx))  # Rz*Ry*Rx
    return R,T

def get_int_params(calib_params):
    int_matrix = [[calib_params['fx'],0,calib_params['cx']],[0,calib_params['fy'],calib_params['cy']],[0,0,1]]
    dist_coefs = [calib_params['k1'],calib_params['k2'],0,0]
    return int_matrix,dist_coefs

def rectify_imgs(R,T,int_params,dist_coefs,  frame_l,frame_r):
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    rectify_scale = 0  # 0=full crop, 1=no crop
    w,h,c = np.shape(frame_l)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(int_params[0], dist_coefs[0],
                                                      int_params[1],
                                                      dist_coefs[1], (1280, 720), # (640, 480),
                                                      R=R, T=T,alpha=rectify_scale)

    left_maps = cv2.initUndistortRectifyMap(int_params[0], dist_coefs[0], R1, P1, (1280, 720) ,# (640, 480)
                                            cv2.CV_16SC2)
    right_maps = cv2.initUndistortRectifyMap(int_params[1],dist_coefs[1], R2, P2, (1280, 720),# (640, 480)
                                             cv2.CV_16SC2)

    left_img_remap = cv2.remap(frame_l, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(frame_r, right_maps[0], right_maps[1],cv2.INTER_LANCZOS4)

    #  draw the images side by side
    total_size = (max(left_img_remap.shape[0], right_img_remap.shape[0]),
                  left_img_remap.shape[1] + right_img_remap.shape[1],3)
    img = np.zeros(total_size, dtype=np.uint8)
    img[:left_img_remap.shape[0], :right_img_remap.shape[1]] = left_img_remap
    img[:right_img_remap.shape[0], left_img_remap.shape[1]:] = right_img_remap
    return left_img_remap,right_img_remap,Q

def calc_disparity(img_l,img_r):
    window_size = 3         ### minDisparity = -100, numDisparities = 160
    left_matcher = cv2.StereoSGBM_create(minDisparity = -100, numDisparities = 160,
                                         blockSize = 3, P1 = 8 * 3 * window_size ** 2,
                                         P2 =0,# 32 * 3 * window_size ** 2,
                                         disp12MaxDiff = 1,
                                         uniquenessRatio = 15, speckleWindowSize = 0, speckleRange = 2,
                                         preFilterCap = 63, mode = cv2.STEREO_SGBM_MODE_HH)#cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print('computing disparity...')
    displ = left_matcher.compute(img_l,img_r)# .astype(np.float32)/16
    dispr = right_matcher.compute(img_l,img_r) # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, img_l, None, dispr) # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    disp_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_AUTUMN)#cv2.COLORMAP_OCEAN)

    return filteredImg, disp_Color

def project_into_3d(Q, depth):

    # get_mesh.main(imgL_BGR=imgL, Q=Q, disp= depth)
    return cv2.reprojectImageTo3D(disparity=depth, Q=Q,)

def ZED_GO():
    global disp2
    global f, base
    global fy, fx, cx, cy

    calib_param_l, calib_param_r, exter_pars = cam_calibrate('HD')
    int_l, dis_l = get_int_params(calib_param_l)
    int_r, dis_r = get_int_params(calib_param_r)
    f, base = calib_param_r['fx'], exter_pars['BaseLine']
    fx, fy, cx, cy = calib_param_r['fx'], calib_param_r['fy'], calib_param_r['cx'], calib_param_r['cy']


    # cap = cv2.VideoCapture(701)
    # cap.set(3, 2560)
    # cap.set(4, 720)

    # while True:
    for file in listdir('Images/test_lake'):

        # flag, frame = cap.read()
        print(file)
        frame = cv2.imread('Images/test_lake/' + (file))
        w, l, c = np.shape(frame)
        frame_l, frame_r = frame[:, :int(l / 2), :], frame[:, int(l / 2):, :]
        R, T = Rectify_Params(exter_pars, mode='HD')

        rect_l, rect_r, Q = rectify_imgs(R=R, T=T, int_params=np.array([int_l, int_r]),
                                         dist_coefs=np.array([dis_l, dis_r]), frame_l=frame_l, frame_r=frame_r)
        # continue
        disp, disp_color = calc_disparity(img_r=rect_r,img_l=rect_l)
        print('Disparity map shape is ', np.shape(disp))
        # plt.show()
        # (img_l=cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY),img_r=cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))

        # disp2 = copy.deepcopy(disp)
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", get_distance)
        # cv2.imshow("image", disp_color)
        # key = cv2.waitKey() & 0xFF
        img_1 = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        # Distance =
        indices = np.argwhere(disp>= -100)
        print(np.shape(indices))
        indices = np.reshape(indices,newshape=(720,1280,2))
        X,Y,Z = XYZ_coordinates( Depth = disp, u = indices[:,:,0], v = indices[:,:,1] )
        # points = project_into_3d(depth=disp, Q=Q)
        # X,Y,Z = points[:,:,0],points[:,:,1],points[:,:,2]
        water_ixds = np.argwhere(Y<-0.1)
        # print(water_ixds)
        # print([Left_pixels[water_ixds,:]])
        print(water_ixds)
        plt.scatter(water_ixds[:, 0],water_ixds[:,1], c='b', s=1)
        plt.imshow(frame_l)
        plt.show()
        idx=0
        font = cv2.FONT_HERSHEY_SIMPLEX
        indices = np.reshape(indices,newshape=(720*1280,2))
        print(np.shape(Z))
        for x,y in tuple(indices):
            if idx%10000==0:
                cv2.rectangle(img_1, (x, y), (x + 2, y + 2), (255, 0, 00), 1)
                cv2.putText(img_1,str(np.round(X[x,y],3)),(x,y-30), font, 0.5,(0,0,255),2,cv2.LINE_AA )
                cv2.putText(img_1,str(np.round(Y[x,y],3)),(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA )
                cv2.putText(img_1,str(np.round(Z[x,y],2)),(x,y+30), font, 0.5,(255,0,0),2,cv2.LINE_AA )
                # cv2.putText(img_1,str(Z[idx]),(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA )
                # idx+=1 \
            idx+=1

        cv2.imshow('left image with Depth annotated',img_1)
        cv2.waitKey(0)

        # if key == ord("p"):
        #     point_cloud = project_into_3d(imgL=frame_l, Q=Q, depth=disp)

class base:
    def __init__(self,imgL,imgR):
        self.imgL = imgL
        self.imgR = imgR
    def cooridnates(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class feature_extraction(base):
    def __init__(self,imgL,imgR):#imgR,imgL):
        super().__init__(imgL,imgR)
        self.feature_extractors = {0:self.ORB(),1:self.SIFT(),2:self.SURF(),3:self.BRISK()
            ,4:self.BRIEF(),5:self.AKAZE(), 6:self.KAZE()}

    def ORB(self):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT

        kp1, des1 = orb.detectAndCompute(self.imgL, None)
        kp2, des2 = orb.detectAndCompute(self.imgR, None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        self.matches = sorted(matches, key=lambda x: x.distance)

        # filtered_matches,
        Left_Pixels, Right_Pixels = self.matcher_to_pixel_coordinates \
            (kp1=kp1, kp2=kp2)

        return Left_Pixels,Right_Pixels

    def SIFT(self):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.imgL, None)
        kp2, des2 = sift.detectAndCompute(self.imgR, None)

        good_R = []
        good_L = []

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # ratio test as per Lowe's paper
        for i, (m1, m2) in enumerate(matches):
            if m1.distance < 0.7 * m2.distance:
                if np.abs(kp1[m1.queryIdx].pt[1] - kp2[m1.trainIdx].pt[1]) < 10:
                    good_L.append(kp1[m1.queryIdx].pt)
                    good_R.append(kp2[m1.trainIdx].pt)


        Right_Pixels = copy.deepcopy(good_R)
        Left_Pixels = copy.deepcopy(good_L)

        return Left_Pixels,Right_Pixels

    def SURF(self):

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SURF_create()#nfeatures=0, edgeThreshold=20)


        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.imgL, None)

        kp2, des2 = sift.detectAndCompute(self.imgR, None)
        # BFMatcher with default params

        good_R = []
        good_L = []
        good_matches = []

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask

        # ratio test as per Lowe's paper
        for i, (m1, m2) in enumerate(matches):
            if m1.distance < 0.8 * m2.distance:
                if np.abs(kp1[m1.queryIdx].pt[1] - kp2[m1.trainIdx].pt[1]) < 10:
                    good_L.append(kp1[m1.queryIdx].pt)
                    good_R.append(kp2[m1.trainIdx].pt)
                    good_matches.append(matches[i])


        Right_Pixels = copy.deepcopy(good_R)
        Left_Pixels = copy.deepcopy(good_L)

        return Left_Pixels,Right_Pixels

    # def FAST(self):
	#
    #     fast = cv2.FastFeatureDetector_create()
    #     kp1 = fast.detect(self.imgL, None)
    #     kp2 = fast.detect(self.imgr, None)
    #     br = cv2.BRISK_create();
    #     kp1, des1 = br.compute(imgL, kp1)  # note: no mask here!
    #     kp2, des2 = br.compute(imgR, None )  # note: no mask here!
	#
	#
	#
	#
    #     good_R = []
    #     good_L = []
    #     good_matches = []
	#
    #     # FLANN parameters
    #     FLANN_INDEX_KDTREE = 0
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     search_params = dict(checks=50)  # or pass empty dictionary
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     print(flann)
    #     matches = flann.knnMatch(des1.astype(cv2.CV_32F), des2.astype(cv2.CV_32F), k=2)
    #     # print(len(matches))
	#
    #     # Need to draw only good matches, so create a mask
    #     matchesMask = [[0, 0] for i in range(len(matches))]
	#
    #     # ratio test as per Lowe's paper
    #     for i, (m1, m2) in enumerate(matches):
    #         if m1.distance < 0.7 * m2.distance:
    #             matchesMask[i] = [1, 0]
    #             good_L.append(kp1[m1.queryIdx].pt)
    #             good_R.append(kp2[m1.trainIdx].pt)
    #             good_matches.append(matches[i])
	#
    #     draw_params = dict(matchColor=(0, 255, 0),
    #                        singlePointColor=(255, 0, 0),
    #                        matchesMask=matchesMask,
    #                        flags=0)
    #     # print(np.shape(good_matches), np.shape(matches), np.shape(matchesMask))
    #     # print(np.shape(kp1), np.shape(kp2))
    #     img3 = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, matches, None, **draw_params)
	#
    #     Right_Pixels = copy.deepcopy(good_R)
    #     Left_Pixels = copy.deepcopy(good_L)
	#
    #     return Left_Pixels, Right_Pixels

    def BRISK(self):
        brisk = cv2.BRISK_create()

        (kp1, des1) = brisk.detectAndCompute(self.imgL, None)
        (kp2, des2) = brisk.detectAndCompute(self.imgR, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        self.matches = sorted(matches, key=lambda val: val.distance)

        Left_Pixels, Right_Pixels = self.matcher_to_pixel_coordinates \
            (kp1=kp1, kp2=kp2)

        return Left_Pixels, Right_Pixels

    def BRIEF(self):
        star = cv2.xfeatures2d.StarDetector_create()
        keyPoints1 = star.detect(self.imgL, None)
        keyPoints2 = star.detect(self.imgR, None)


        # Create the BRIEF extractor and compute the descriptors
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        (kp1, des1) = brief.compute(self.imgL, keyPoints1)
        (kp2, des2) = brief.compute(self.imgR, keyPoints2)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.matches = bf.match(des1, des2)

        self.matches = sorted(self.matches, key=lambda val: val.distance)

        Left_Pixels, Right_Pixels = self.matcher_to_pixel_coordinates \
            (kp1=kp1, kp2=kp2)

        return Left_Pixels, Right_Pixels

    def AKAZE(self):
        # Initiate ORB detecto

        akaze = cv2.AKAZE_create()
        kpts1, desc1 = akaze.detectAndCompute(self.imgL, None)
        kpts2, desc2 = akaze.detectAndCompute(self.imgR, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(desc1, desc2, 2)
        matched1 = []
        matched2 = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio

        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                if np.abs(kpts1[m.queryIdx].pt[1] - kpts2[m.trainIdx].pt[1]) < 10:
                    matched1.append(kpts1[m.queryIdx].pt)
                    matched2.append(kpts2[m.trainIdx].pt)

        return matched1, matched2

    def KAZE(self):
        # Initiate ORB detect
        akaze = cv2.KAZE_create()
        kpts1, desc1 = akaze.detectAndCompute(self.imgL, None)
        kpts2, desc2 = akaze.detectAndCompute(self.imgR, None)
        # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        # FLANN parameters


        good_R = []
        good_L = []
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)

        # ratio test as per Lowe's paper
        for i, (m1, m2) in enumerate(matches):
            if m1.distance < 0.7 * m2.distance:
                if np.abs(kpts1[m1.queryIdx].pt[1] - kpts2[m1.trainIdx].pt[1]) < 10:
                    good_L.append(kpts1[m1.queryIdx].pt)
                    good_R.append(kpts2[m1.trainIdx].pt)



        Right_Pixels = copy.deepcopy(good_R)
        Left_Pixels = copy.deepcopy(good_L)

        return Left_Pixels, Right_Pixels

    def matcher_to_pixel_coordinates(self,kp1, kp2):
        '''
        1. iterate through both lists and look for the y axis
        2. if the difference between both y values are > 20 then delete the index
        3. return the filtered matches
        '''
        list_kp1 = [kp1[mat.queryIdx].pt for mat in self.matches]
        list_kp2 = [kp2[mat.trainIdx].pt for mat in self.matches]
        remove_indexes = []

        for idx in range(len(list_kp1)):
            if np.abs(list_kp1[idx][1] - list_kp2[idx][1]) > 10:
                remove_indexes.append(idx)

        ###### delete unstable features
        for ele in sorted(remove_indexes, reverse=True):
            del self.matches[ele] ## filtered matches
            del list_kp1[ele]
            del list_kp2[ele]

        return list_kp1, list_kp2

def Disparity_from_feature(Left_pixels,Right_pixels):
    '''
    :param Left_pixels:
    :param Right_pixels:
    :return: Left image with Depth measurment on top of the features
    '''
    disparity = np.subtract(np.array(Left_pixels),np.array(Right_pixels))

    return disparity

def XYZ_coordinates(Depth,u,v):
    '''

    :param Depth:
    :return:XYZ
    Math:   X = Z / fx * (u - cx)
            Y = Z / fy * (v - cy)
            Z=Depth
    '''
    Distance = (f * base * 0.001) / Depth[:,0] ## 0 so we get for left image

    X = Distance/(fx)*(u-cx)
    Y = Distance/(fy) * (v - cy)
    Z = Distance
    return X,Y,Z

def feature_match_and_project(frame_l,frame_r):
    '''
    Water Cue from Depth
    '''
    start = time.time()
    global f,base
    global fy,fx,cx,cy
    calib_param_l, calib_param_r, exter_pars = cam_calibrate('HD')
    f, base = calib_param_r['fx'], exter_pars['BaseLine']
    fx,fy,cx,cy = calib_param_r['fx'],calib_param_r['fy'],calib_param_r['cx'],calib_param_r['cy']
    int_l, dis_l = get_int_params(calib_param_l)
    int_r, dis_r = get_int_params(calib_param_r)

    # idx=0
    R, T = Rectify_Params(exter_pars, mode='HD')

    rect_l, rect_r, Q = rectify_imgs(R=R, T=T, int_params=np.array([int_l, int_r]),
                                     dist_coefs=np.array([dis_l, dis_r]), frame_l=frame_l, frame_r=frame_r)

    img_1 = cv2.cvtColor(rect_l, cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(rect_r, cv2.COLOR_RGB2GRAY)
    ground_idxs = []
    # data_struct = base()
    extract = feature_extraction(imgL=img_1, imgR=img_2)
    for key in extract.feature_extractors.keys():
        Left_pixels,Right_pixels = extract.feature_extractors[key]
        Distance = Disparity_from_feature(Left_pixels=Left_pixels,Right_pixels=Right_pixels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        l = len(Left_pixels)
        Left_pixels = np.array(Left_pixels, dtype=int)#.reshape(l,2)
        Right_pixels = np.array(Right_pixels, dtype=int)#.reshape(l,2)
        Distance = np.array(Distance)
        X,Y,Z = XYZ_coordinates(Distance,u=Left_pixels[:,0],v=Left_pixels[:,1])

            ########## Draw Distance to objects on Left Image ##############
        water_ixds = np.argwhere(Y<-2.0)
        # ground_idxs.append(water_ixds)

        plt.scatter(Left_pixels[water_ixds,0], Left_pixels[water_ixds,1], c='r', s=10)
        # plt.scatter(Left_pixels[trees_idxs,0], Left_pixels[trees_idxs,1], s=10, c='black')
    end = time.time()
    print('Time it took for depth calulations is ', end - start)
    plt.scatter(Left_pixels[water_ixds, 0], Left_pixels[water_ixds, 1], c='r', s=10, label='<-2.m depth')
    # plt.scatter(Left_pixels[trees_idxs, 0], Left_pixels[trees_idxs, 1], s=10, c='black',
    #             label='Greater than 7 m height')
    # plt.legend(loc='lower left')
    # plt.imshow(rect_l)
    # plt.subplot(211)
    # plt.imshow(rect_l)
    # # plt.figure()
    # # plt.imshow(rect_l)
    # plt.show()
    return rect_l ,water_ixds

def fusing_algo(color_idx,texture_idx):
    start = time.time()
    strong_water_indexes = []
    weak_water_idxs = []
    # print(texture_idx[:10])
    for x,y in  color_idx:
        if [x,y] in texture_idx:
            strong_water_indexes.append([x,y])
        else:
            weak_water_idxs.append([x,y])
    print(np.shape(weak_water_idxs))
    print('Time it took for indexes fusing is ',time.time()-start)
    return strong_water_indexes,weak_water_idxs

def fusion(c_mask,v_mask):
    start = time.time()

    if np.shape(c_mask)!=np.shape(v_mask):
        raise NameError('You cannot Fuse unmatched masks sizes')

    c_mask = np.array(c_mask, dtype=np.uint8)
    v_mask = np.array(v_mask, dtype=np.uint8)
    #### convert them into opencv shape
    img1_bg = cv2.bitwise_and(c_mask, c_mask, mask=v_mask)
    idxs = np.argwhere(img1_bg == True)
    end = time.time()

    print('time it took to fuse was ',end-start)
    return img1_bg, idxs

def main(mode, frame_l,frame_r):
    if mode == 'Local Method':
        rect_l, water_ixds = feature_match_and_project(frame_l=frame_l,frame_r=frame_r)
        return rect_l ,water_ixds
    elif mode == 'Semi Global Method':
        ZED_GO()
        return  None , None