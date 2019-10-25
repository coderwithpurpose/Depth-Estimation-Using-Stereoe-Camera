import cv2
import os
import numpy as np

refPt = []
cropping = False
def crop_img (image,idxw,idxg):
	def click_and_crop(event, x, y, flags, param):
	    # grab references to the global variables
	    global refPt, cropping

	    # if the left mouse button was clicked, record the starting
	    # (x, y) coordinates and indicate that cropping is being
	    # performed
	    if event == cv2.EVENT_LBUTTONDOWN:
	        refPt = [(x, y)]
	        cropping = True

	    # check to see if the left mouse button was released
	    elif event == cv2.EVENT_LBUTTONUP:
	        # record the ending (x, y) coordinates and indicate that
	        # the cropping operation is finished
	        refPt.append((x, y))
	        cropping = False

	        # draw a rectangle around the region of interest
	        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
	        cv2.imshow("image", image)

	# construct the argument parser and parse the arguments

	# load the image, clone it, and setup the mouse callback function
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	# keep looping until the 'q' key is pressed
	while True:
	    # display the image and wait for a keypress
	    cv2.imshow("image", image)
	    key = cv2.waitKey(1) & 0xFF

	    # ireset the cropping region
	    if key == ord("r"):
	        image = clone.copy()
	        continue
		# go to next image
	    elif key == ord("q"):
		    cv2.destroyAllWindows()
		    return idxw,idxg


	    elif key == ord("w"):
		    if len(refPt) == 2:
			    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			    # cv2.imshow("ROI", roi)
			    cv2.imwrite('cropped_water/' + str(idxw).zfill(3) + '.png', roi)
			    idxw += 1
			    # cv2.waitKey(0)

	    # saved cropped region
	    elif key == ord("g"):
		    if len(refPt) == 2:
			    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			    # cv2.imshow("ROI", roi)
			    cv2.imwrite('cropped_grass/'+str(idxg).zfill(3)+'.png',roi)
			    idxg+=1
			    # cv2.waitKey(0)

	    # kill cropping process
	    elif key == ord("k"):
		    cv2.destroyAllWindows()
		    break

	cv2.destroyAllWindows()
if __name__ == '__main__':
	idxw,idxg = 0,0
	for filename in os.listdir('../outdoor_may24'):
		img = cv2.imread('../outdoor_may24/'+filename)
		_, l, __ = np.shape(img)
		imgl, imgr = img[:, :int(l / 2), :], img[:, int(l / 2):, :]
		idxw,idxg = crop_img(image=imgl,idxw=idxw,idxg=idxg)
		idxw+=1
		idxg+=1
