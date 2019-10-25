import os
from skimage import io
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from main import *


'''
return 4 coloum matrices 
col1: Sat/Value 
col2: Var calculated over saturation channel 
col3: Var calculated over value channel 
col4: labels with 1 means water detected and 0 means grass detected 
'''

def pre_process():
	v_arr = np.array([])
	s_arr =  np.array([])
	h_arr =  np.array([])

	v_arr_b = np.array([])
	s_arr_b = np.array([])
	h_arr_b = np.array([])

	sv_arr = np.array([])
	hs_arr = np.array([])
	hv_arr = np.array([])

	for filename in os.listdir('cropped_grass'):  # Lake_14th_may/Low_position
		print(filename)
		img = io.imread('cropped_grass/' + filename)  # test_lake
		_, l, __ = np.shape(img)

		gaus_img = ndimage.gaussian_filter(img, sigma=(5, 5, 0), order=0)
		space = int((5 - 1) / 2)

		#     return mask_sv[space:-space,space:-space], mask_hs[space:-space,space:-space],mask_hv[space:-space,space:-space]
		sv, hs,hv = Hue_Filtering(gaus_img=gaus_img,space=space)
		# return Sat_img[space:-space, space:-space], Value_img[space:-space, space:-space],Hue_img[space:-space, space:-space]
		s_var,v_var,h_var = Calc_Var(original_image=img,var_threshold=20, window=5)
		s_var_b,v_var_b,h_var_b = Calc_Var(original_image=gaus_img,var_threshold=20, window=5)

		####   Var over the 3 HSV channels arrays######
		v_arr  = np.hstack((v_arr,v_var.flatten()))
		s_arr  = np.hstack((s_arr,s_var.flatten()))
		h_arr  = np.hstack((h_arr,h_var.flatten()))
		### same VAR with Blur preprocess #######
		v_arr_b = np.hstack((v_arr_b, v_var_b.flatten()))
		s_arr_b = np.hstack((s_arr_b, s_var_b.flatten()))
		h_arr_b = np.hstack((h_arr_b, h_var_b.flatten()))
		#### HSV color arrays #####
		sv_arr = np.hstack((sv_arr, sv.flatten()))
		hs_arr = np.hstack((hs_arr, hs.flatten()))
		hv_arr = np.hstack((hv_arr, hv.flatten()))
#
#
#
#
# # sv_Arr  = np.hstack((sv_Arr,sv.flatten()))
# #
# # 	water = np.vstack((sv_Arr,var_arr))
# # 	print(np.shape(water))
# # 	np.save('grass_fuse.npy', water)
#
	np.save('grass_h_var_arr',h_arr)
	np.save('grass_v_var_arr',v_arr)
	np.save('grass_s_var_arr',s_arr)

	np.save('grass_h_b_var_arr',h_arr_b)
	np.save('grass_v_b_var_arr',v_arr_b)
	np.save('grass_s_b_var_arr',s_arr_b)

	np.save('grass_sv_arr',sv_arr)
	np.save('grass_hs_arr',hs_arr)
	np.save('grass_hv_arr',hv_arr)


def prepare_for_ML():

	water = np.vstack((np.load('Data/run2/water/water_sv_arr.npy'),np.load('Data/run2/water/water_hs_arr.npy'),np.load('Data/run2/water/water_hv_arr.npy'),
	                  np.load('Data/run2/water/water_s_b_var_arr.npy'),np.load('Data/run2/water/water_v_b_var_arr.npy'),np.load('Data/run2/water/water_h_b_var_arr.npy'),
	                  np.load('Data/run2/water/water_s_var_arr.npy'),np.load('Data/run2/water/water_v_var_arr.npy'),np.load('Data/run2/water/water_h_var_arr.npy')))

	grass = np.vstack((np.load('Data/run2/grass/grass_sv_arr.npy'), np.load('Data/run2/grass/grass_hs_arr.npy'),np.load('Data/run2/grass/grass_hv_arr.npy'),
	                  np.load('Data/run2/grass/grass_s_b_var_arr.npy'),np.load('Data/run2/grass/grass_v_b_var_arr.npy'),np.load('Data/run2/grass/grass_h_b_var_arr.npy'),
	                  np.load('Data/run2/grass/grass_s_var_arr.npy'), np.load('Data/run2/grass/grass_v_var_arr.npy'),np.load('Data/run2/grass/grass_h_var_arr.npy')))

	grass = grass.transpose()
	water = water.transpose()
	np.save('Data/run2/grass_dataset',grass)
	np.save('Data/run2/water_dataset',water)

	# print(np.shape(grass))
	# print(np.shape(water))
	# print(grass)
	y_g = np.zeros(len(grass))
	l = len(y_g)
	y_g.reshape((l, 1))
	# print(y_g.shape)
	y_w = np.ones(len(water))
	y_w.reshape((len(y_w), 1))

	x = np.vstack((water, grass))
	y = np.hstack((y_w, y_g))
	y = y.reshape((len(y), 1))
	print(y.shape)

	data = np.hstack((x, y))
	print(data.shape)
	np.random.shuffle(data)

	X = data[:, :-1]
	Y = data[:, -1]

	return data
if __name__ == '__main__':
	data = prepare_for_ML()
	np.save('Data/run2/DATASET',data)
	print(data[:,-1])