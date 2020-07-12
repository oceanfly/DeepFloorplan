import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image
from resizeimage import resizeimage
import cv2
from imageio import imread, imsave
from matplotlib import pyplot as plt
sys.path.append('/Users/taosun/Documents/GitHub/DeepFloorplan/utils/')
from util import *
from rgb_ind_convertor import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='/Users/taosun/Documents/GitHub/DeepFloorplan/demo/test12.png',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255, 255, 255],  # background
	# 1: [192, 192, 224],  # closet
	1: [255, 224, 128],  # closet -> bedroom
	2: [192, 255, 255],  # batchroom/washroom
	3: [224, 255, 192],  # livingroom/kitchen/dining room
	4: [255, 224, 128],  # bedroom
	# 5: [255, 160, 96],   # hall
	5: [255, 224, 128],   # hall -> bedroom
	6: [255, 224, 224],  # balcony
	7: [255, 255, 255],  # not used
	8: [255, 255, 255],  # not used
	9: [255, 60, 128],  # door & window
	10: [0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im == i)] = rgb

	return rgb_im


def main(args):
	# load input
	im = cv2.imread(args.im_path)
	im = cv2.resize(im, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	# cv2.imshow('sample', im)
	# cv2.waitKey(0)
	# im = cv2.fastNlMeansDenoisingColored(im,None,21,21,7,21)

	for i in range(512):
		for j in range(512):
			pixel_RGB = im[i,j]
			if np.amax(pixel_RGB) == 0:
				im[i,j] = [255,255,255] # white
			elif pixel_RGB[2] > pixel_RGB[0]:
				im[i,j] = [255,255,255] # white	
			else:
				im[i,j] = [0,0,0]  # black
	
	im_mask = im / 255
	filter_h = np.array([[0,0,0],
						[1,1,1],
						[0,0,0]], dtype=np.uint8)				
	im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_CLOSE, filter_h)
	filter_v = np.array([[0],
						[1],
						[0]], dtype=np.uint8)				
	im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_CLOSE, filter_v)

	door_list = []

	for i in range(512):
		row = im_mask[i]
		# print("row:",row.shape)
		p1 = 0
		p2 = 0
		for j in range(512):
			if np.amax(row[j]) == 0: # find a black dot
				# print("row j:",row[j])
				if p1 == 0:
					p1 = j
				elif j - p1 > 1: 
					p2 = j
				else:
					p1 = j
			if p2 - p1 > 15:
				if p2 - p1 < 30:
					print("i=",i)
					print("p1=",p1)
					print("p2=",p2)
					door_list.append([i, p1, p2-p1])
					for c in range(p1+1,p2):
						#  print("im_mask=",im_mask[i,c])
						im_mask[i,c] = [0,0,0]
					p1 = 0
				else:
					p1 = p2
				
	print("door=", door_list)
		
	im = im_mask * 255
	cv2.imshow('sample', im_mask)
	cv2.waitKey(0)
	kernel = np.ones((1,2),np.uint8)	
	im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((2,1),np.uint8)
	im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

	# create tensorflow session
	with tf.compat.v1.Session() as sess:
		
		# initialize
		sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
					tf.compat.v1.local_variables_initializer()))

		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('/Users/taosun/Documents/GitHub/DeepFloorplan/pretrained/pretrained_r3d.meta')
		saver.restore(sess, '/Users/taosun/Documents/GitHub/DeepFloorplan/pretrained/pretrained_r3d')

		# get default graph
		graph = tf.compat.v1.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		# print("floorplan=", floorplan.shape)
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan, color_map=floorplan_fuse_map) / 255


		save_dir = "/Users/taosun/Documents/GitHub/DeepFloorplan/suzhou/output/test12.png"
		imsave(save_dir, floorplan_rgb)



		# plot results
		plt.subplot(121)
		plt.imshow(im)
		plt.subplot(122)
		plt.imshow(floorplan_rgb)
		plt.show()

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
