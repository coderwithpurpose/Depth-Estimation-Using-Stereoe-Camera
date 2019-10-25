import numpy as np
import matplotlib.pyplot as plt


def plot_plz(nump_arr):
	nump_arr = np.sort(nump_arr)
	print(np.amax(nump_arr))
	plt.plot(nump_arr)
	plt.ylabel('some numbers')
	plt.show()
	pass

def draw_hist(nump_arr):

	hist, bins = np.histogram(nump_arr, bins=10)
	# width = 0.7 * (bins[1] - bins[0])
	plt.figure()
	width = np.diff(bins)
	center = (bins[:-1] + bins[1:]) / 2
	# plt.bar(center, hist, align='center', width=width)
	return hist

def both_hists(grass,water):
	from matplotlib import pyplot
	from scipy import stats
	bins = np.linspace(0, 10000, 100)
	# y = stats.norm.pdf(water, loc=np.mean(water),scale=np.std(water))
	pyplot.hist(grass, bins, alpha=0.5, label='grass',color='green')
	# pyplot.hist(y, bins, alpha=0.5, label='grass',color='red')
	# pyplot.hist(water, bins, alpha=0.5, label='water', color='blue')
	pyplot.legend(loc='upper right')
	pyplot.title('Distribution of Grass vs Water')
	pyplot.ylabel('counts', fontsize=16)
	pyplot.xlabel('Saturation Intensity Variance', fontsize=16)
	# pyplot.show()
	plt.figure()
	bins = np.linspace(0, 60, 100)
	pyplot.hist(water, bins, alpha=0.5, label='water', color='blue')
	pyplot.legend(loc='upper right')
	pyplot.title('Distribution of Grass vs Water')
	pyplot.ylabel('counts', fontsize=16)
	pyplot.xlabel('Saturation Intensity Variance', fontsize=16)
	pyplot.show()

def plot_both(pdf,water):
	plt.plot(pdf)
	plt.plot(water)
	plt.show()


def half_normal(g,w):

	# std = np.std(g)
	# y= (np.sqrt(2)/(std*np.sqrt(np.pi)))*std*np.exp(-g**2/(2*std*std))
	# std2 = np.std(g)
	# x = (np.sqrt(2) / (std2 * np.sqrt(np.pi))) * std2 * np.exp(-w ** 2 / (2 * std2 * std2))
	y = g
	x = w
	# both_hists(y,x)
	plt.plot(y, label='grass')
	plt.plot(x, label='water')
	plt.legend(loc='upper right')

	return y

def multiple_hists(x1,x2,x3):
	plt.figure()
	plt.hist(x1, color='g', label='Hue')
	plt.hist(x2,  color='b', label='Sat')
	plt.hist(x3, color='r', label='val')
	plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')
	# plt.xlim(50, 75)
	plt.legend();


def twohists(x1,x2,title):
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	n, bins, patches = ax1.hist(x1)
	ax1.set_xlabel('Angle a (degrees)')
	ax1.set_ylabel('Frequency')

	n, bins, patches = ax2.hist(x2)
	ax2.set_xlabel('Angle b (degrees)')
	ax2.set_ylabel('Frequency')


	# plt.figure()
	# plt.hist(x1, color='g', label='grass')
	# plt.hist(x2,  color='b', label='water')
	# plt.gca().set(title=title, ylabel='Frequency')
	# plt.xlim(0, 2000)
	# plt.legend();

if __name__ == '__main__':

	# data= np.load('Data/run2/DATASET.npy')
	keys  = np.load('Data/run2/keys.npy',allow_pickle=True).flat[0]
	grass = np.load('Data/run2/grass_dataset.npy')
	water = np.load('Data/run2/water_dataset.npy')
	grass = grass.transpose()
	water = water.transpose()
	print((np.shape(grass[0]),np.shape(water[0])))
	# print((keys[0]))
	for i in range(0,9):
		twohists(grass[i],water[i],keys[i])
	plt.show()

# for i in range (0,len(grass)):
	# 	pass
	# print(np.shape(grass))
	# print(np.shape(water))
	# keys= {0:'sv',1:'hs',2:'hv',3:'s_b_var',4:'v_b_var',5:'h_b_var',6:'s_var', 7:'v_var_arr.npy',8:'h_var_arr.npy'}
	# np.save('keys',keys)
	# g_v_arr = np.load('Data/grass_v_var_arr.npy')
	# g_s_arr = np.load('Data/grass_s_var_arr.npy')
	# g_h_arr = np.load('Data/grass_h_var_arr.npy')
	# w_v_arr = np.load('Data/water_v_var_arr.npy')
	# w_s_arr = np.load('Data/water_s_var_arr.npy')
	# w_h_arr = np.load('Data/water_h_var_arr.npy')

	# twohists(g_h_arr,w_h_arr,title='its ok')
	# twohists(g_s_arr,w_s_arr,title='histograms of variance calculated over saturation channel')
	# twohists(g_v_arr,w_v_arr, title='histograms of variance calculted over Value channel')
	# plt.show()



