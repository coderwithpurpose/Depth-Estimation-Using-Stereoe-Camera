import numpy as np
import cv2
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from os import listdir
import  generate_pointcloud as get_mesh
import copy
import os
from skimage.color import rgb2gray
# from skimage.feature import match_descriptors, ORB, plot_matches , daisy
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import pptk
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
		Distance = (f * base * 0.001) / disp2[y, x]
		# Distance= np.around(Distance*0.001,decimals=3)
		print('Distance: ' + str(Distance) + ' m')


class camera_info:
    def __init__(self):
        self.Q = None
        self.T = None
        self.R = None
        pass



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
