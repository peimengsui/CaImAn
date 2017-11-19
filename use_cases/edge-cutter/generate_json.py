import os
import numpy as np
import pickle
import json
from skimage import io, img_as_ubyte
from skimage.color import grey2rgb

load_dir = '/mnt/ceph/neuro/edge_cutter/1_input_data'
run_dir = '/mnt/home/speimeng/dev/TensorBox/data/neuron_one'
img_filelist = ['Yr_d1_512_d2_512_d3_1_order_C_frames_2936_..npz']
bounding_box_border = 3
json_list = []

for img_filename in img_filelist:
	frames = np.load(os.path.join(load_dir, img_filename))['arr_0']
	labels = pickle.load(open(os.path.join(load_dir, img_filename[:-3]+'pkl'),"rb"))
	num_frames = frames.shape[0]
	for i in range(num_frames):
		dic = {}
		dic['image_path'] = 'train_frames/'+img_filename[:-5]+str(i)+'.png'
		image = frames[i,:,:]
		#image = img_as_ubyte(grey2rgb(image))
		io.imsave(os.path.join(run_dir, dic['image_path']), image)
		dic['rects'] = []
		boxes = labels[i]
		for rect in boxes:
			coord = {}
			coord['x1'] = float(rect[1])-bounding_box_border
			coord['x2'] = float(rect[1])+bounding_box_border
			coord['y1'] = float(rect[0])-bounding_box_border
			coord['y2'] = float(rect[0])+bounding_box_border
			dic['rects'].append(coord)
		json_list.append(dic)
with open(os.path.join(run_dir,'neuron_train_boxes.json'), 'w') as outfile:
    json.dump(json_list, outfile)
