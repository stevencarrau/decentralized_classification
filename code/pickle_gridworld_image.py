from skimage import io
import cv2
import pickle
import pathlib

def write_pickled_image(filename):
	data = io.imread(filename[0])
	data = cv2.resize(data, filename[1], interpolation=filename[2])
	pickle.dump(data,open("Gridworld.p","wb"), protocol=2)


env_loc = pathlib.Path().absolute()
# mapname = 'RVR_2_7_20_site_cropped'
# scale = (74,29)

# Create and load map
mapname = 'rvr_2020_08_03_site_controller'
scale = (63,24)
fname = [str(env_loc)+'/'+mapname+'.png',scale,cv2.INTER_LINEAR_EXACT]
write_pickled_image(fname)