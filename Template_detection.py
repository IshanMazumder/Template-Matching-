import cv2
import numpy as np

from sys import argv

def template_match(spath,lpath,algo_type):
	
	method = cv2.TM_SQDIFF_NORMED
	small_image = cv2.imread(spath, cv2.IMREAD_GRAYSCALE)
	large_image = cv2.imread(lpath,cv2.IMREAD_GRAYSCALE)
	global s, l , c
	size = 1
	width_s = int(small_image.shape[1]*size)
	height_s = int(small_image.shape[0]*size)

	#print(large_image.shape)
	width_l = int(large_image.shape[1]*size)
	height_l = int(large_image.shape[0]*size)

	small_image = cv2.resize(small_image, (width_s,height_s),interpolation= cv2.INTER_AREA)
	
	large_image = cv2.resize(large_image, (width_l,height_l),interpolation= cv2.INTER_AREA)
	w, h = small_image.shape[::-1]
	
	kernal = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	small_image = cv2.filter2D(small_image, -1, kernal)
	large_image = cv2.filter2D(large_image, -1, kernal)
	
	k = np.ones((5,5), np.uint8)
	
	if algo_type == "canny":
		_,s_bin_inv = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		_,l_bin_inv = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		s_canny_edge = cv2.Canny(s_bin_inv,50,300,L2gradient= False)
		l_canny_edge = cv2.Canny(l_bin_inv,50,300,L2gradient= False)
		s = s_canny_edge
		l = l_canny_edge
		c=1
	elif algo_type == "gradient": 
		_,s_bin_inv = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		#_,l_bin_inv = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		
		s_eroded_minus_dilated = cv2.morphologyEx(s_bin_inv, cv2.MORPH_GRADIENT, k)
		#l_eroded_minus_dilated = cv2.morphologyEx(l_bin_inv, cv2.MORPH_GRADIENT, k)
		s = s_eroded_minus_dilated
		l = large_image
		c=2
	elif algo_type == "bin_inv":
		_,s_bin_inv = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		_,l_bin_inv = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		s = s_bin_inv
		l = l_bin_inv
		c=3
	elif algo_type == "all":
		_,s_bin_inv = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		_,l_bin_inv = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		s_canny_edge = cv2.Canny(s_bin_inv,50,300,L2gradient= False)
		l_canny_edge = cv2.Canny(l_bin_inv,50,300,L2gradient= False)
		s_eroded_minus_dilated = cv2.morphologyEx(s_canny_edge, cv2.MORPH_GRADIENT, k)
		l_eroded_minus_dilated = cv2.morphologyEx(l_canny_edge, cv2.MORPH_GRADIENT, k)
		s = s_eroded_minus_dilated
		l = l_eroded_minus_dilated
	elif algo_type == "lap":
		_,s_bin_inv = cv2.threshold(small_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		_,l_bin_inv = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		s_lap = cv2.Laplacian(s_bin_inv, -1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
		l_lap = cv2.Laplacian(l_bin_inv, -1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
		s = s_lap
		l = l_lap
	else:
		s = small_image
		l = large_image
		c=0


	try:
		result = cv2.matchTemplate(s, l, method)
		min_val, max_val, min_Loc, max_Loc = cv2.minMaxLoc(result)
		x, y = min_Loc
		#x1 = x+w
		#y1 = y+h
		
	except:
			#print("the image didnt match")
			return False, 0,0,0,0
			
	else:
		if x != 0 and y != 0:
			#print("the images matched")
			#w = int((x+w)/2)
			#h = int((y+h)/2)
			return True, x, y, w,h
		else:
			#print("the image didnt match")
			return False, 0,0,0,0

# python Template_detection.py path/to/file1 path/to/file2
#if __init__ == "__main__":

first_file = argv[1]
second_file = argv[2]
algo_type = argv[3]

res, x, y,w,h = template_match(first_file, second_file, algo_type)
print(f'{res} {x} {y} {w} {h}')
