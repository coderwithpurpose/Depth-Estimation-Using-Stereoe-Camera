import numpy as np
import cv2
import matplotlib.pyplot as plt


def imshow_components(labels,img,idx):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

    labeled_img[label_hue==0] = 0
    idxs = np.argwhere(labeled_img>0)
    plt.figure()
    plt.scatter(idxs[:, 1], idxs[:, 0], c='blue', s=1, label='Fusion Cue') 
    plt.imshow(img)
    # plt.show()
    plt.savefig('w_area/w_area_'+str(idx).zfill(2)+'.png')

#     plt.imshow(img)
#     plt.show()
    
def plot_mask(img,mask,idx):
	idxs = np.argwhere(mask == 1)
	plt.figure()
	plt.scatter(idxs[:, 1], idxs[:, 0], c='blue', s=1, label='Fusion Cue')
	plt.imshow(img)
	# plt.show
	plt.savefig('w_area/w_orig_'+str(idx).zfill(2)+'.png')
idx=0



def plot_all(labels,img,mask,idx):
	label_hue = np.uint8(179 * labels / np.max(labels))
	blank_ch = 255 * np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

	labeled_img[label_hue == 0] = 0
	idxs = np.argwhere(labeled_img > 0)
	# plt.figure()
	# plt.scatter(idxs[:, 1], idxs[:, 0], c='blue', s=1, label='Fusion Cue')
	# plt.imshow(img)
	# plt.show()
	# plt.savefig('w_area/w_area_' + str(idx).zfill(2) + '.png')

	m_idxs = np.argwhere(mask == 1)
	# plt.figure()
	# plt.scatter(idxs[:, 1], idxs[:, 0], c='blue', s=1, label='Fusion Cue')
	# plt.imshow(img)
	plt.figure()
	plt.subplot(3, 1,1)
	plt.imshow(img)
	plt.title('original image')


	plt.subplot(3, 1,2)

	plt.scatter(m_idxs[:, 1], m_idxs[:, 0], c='blue', s=1, label='Fusion Cue')
	plt.imshow(img)
	plt.title('water_detection')
	plt.imshow(blobs_labels, cmap='nipy_spectral')
	plt.axis('off')

	plt.tight_layout()
	plt.show()

	plt.subplot(3, 1,3)
	plt.scatter(idxs[:, 1], idxs[:, 0], c='blue', s=1, label='Fusion Cue')
	# plt.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')

	plt.imshow(img)
	plt.title('water_detection_with_area_filtering')
	# plt.savefig('w_area/w_area_' + str(idx).zfill(2) + '.png')
	plt.show()


for x in range(0,11):    
    mask1 = np.load('down_sampled_imgs/masks/mask_'+str(x).zfill(2)+'.npy')
    print('original shape of the mask is ', np.shape(mask1))
    mask = cv2.blur(mask1,(51,51))

    img = cv2.imread('down_sampled_imgs/img_'+str(x).zfill(2)+'.png') #     RGB_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w,l = np.shape(mask)

    for xidx in range(0,120): 
        for yidx in range(0,120):
            if(xidx+yidx)<121: 
                mask[xidx,yidx] = 0 
                mask[xidx,-yidx] = 0 

    output = cv2.connectedComponentsWithStats(mask,connectivity=4)

    num_labels = output[0] 
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    '''
Statistics output for each label, including the background label, see below for available statistics. 
Statistics are accessed via stats[label, COLUMN] where available columns are defined below.

cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding
box in the horizontal direction.
cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box 
in the vertical direction.
cv2.CC_STAT_WIDTH The horizontal size of the bounding box
cv2.CC_STAT_HEIGHT The vertical size of the bounding box
cv2.CC_STAT_AREA The total area (in pixels) of the connected component
    '''
    # The fourth cell is the centroid matrix
    centroids = output[3]

    filtered = 0 
    print(np.unique(stats[:,-1]))

    print('there area %s water bodies' %num_labels)
    for i in range(0,num_labels): 
        area = stats[i,-1]
        print(' area of body number {} is {}'.format(i,area))
        if area < 800: 
            idxs = np.argwhere(labels==i)
            labels = np.transpose(labels)
            labels[idxs] = 0  
            labels= np.transpose(labels) 
            filtered+=1 
        else: 
            font = cv2.FONT_HERSHEY_COMPLEX  
            x,y=((centroids[i]))
            cv2.putText(labels, str(area) , (int(x),int(y)), font, 1, (0, 255, 0))
    # plot_mask(img,mask1,idx)
    # imshow_components(labels,img,idx)
    plot_all(labels=labels,img=img,mask=mask1,idx=idx)

    idx += 1

