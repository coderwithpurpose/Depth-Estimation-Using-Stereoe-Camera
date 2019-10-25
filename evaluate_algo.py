import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(mat):
	min = np.unique(mat)[0]
	mask = (mat>min)
	return mask

def prepare_data():
	data_set = []
	gtruth = []
	for filename in os.listdir('Images/LabelMe_DataSet/labels'):
	    # img = plt.imread('Images/LabelMe_DataSet/PixelLabelData_1/'+filename)
	    path = 'Images/LabelMe_DataSet/labels/'+filename
	    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	    data = preprocess(img)
	    gtruth.append(data)

	for filename in os.listdir('Images/scatter_pts_only_results'):
	    # img = plt.imread('Images/scatter_pts_only_results/'+filename)
	    path = 'Images/scatter_pts_only_results/'+filename
	    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	    data = preprocess(img)
	    data_set.append(data)
	print(np.shape(gtruth))
	print(np.shape(data_set))
	return gtruth,data_set

def eval(labels, prediction):
	l,height,width = np.shape(labels)
	mask = []
	for i in range(l):
		mask.append(labels[i] == prediction[i])
	mask = (np.reshape(mask,(l,height*width)))
	# print(np.shape(mask))
	print('accuracy of the model is ',round((len(np.argwhere(mask==True))/(l*height*width)*100),2))
	stats(labels,prediction)

def stats(labels,prediction):
	l,height,width = np.shape(labels)
	t_p,f_p,f_n,t_n = 0,0,0,0
	for i in range (l):
		for x in range(height):
			for y in range(width):
				if labels[i][x,y] == prediction[i][x,y] == True:
					t_p+=1
				elif labels[i][x,y] != prediction[i][x,y] and prediction[i][x,y] ==True :
					f_p+=1
				elif labels[i][x,y] == prediction[i][x,y] == False:
					t_n+=1
				elif labels[i][x,y] != prediction[i][x,y] and  prediction[i][x,y] == False:
					f_n+=1
	size = 	l*height*width
	# print(t_p,t_n,f_p,f_n)
	print('tp acc is'+str(round(t_p/size*100.0,2)),'tn acc is'+ str(round(t_n/size*100.0,2)),
	      'fp acc is'+str(round(f_p/size*100.0,2)),'fn acc is'+str(round(f_n/size*100.0,2)))
	print('number os samples is',str(round(size/10**6,2)),' Million')

if __name__ == '__main__':
    labels, data = prepare_data()
    eval(labels=labels,prediction=data)

