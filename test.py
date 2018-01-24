import scipy.misc
import numpy as np
import cv2
import imageio as io

import tifffile as tiff

path1 = './Test/CT/usb.tiff'

path2 = './Test/Set5/bird_GT.bmp'
path3 = './Test/CT/man.png'
ret1 = scipy.misc.imread(path3).astype(np.float)
#ret1 = cv2.imread(path1, flags = -1).astype(np.float)
#ret2 = io.imread(path1).astype(np.float)
#ret3 = scipy.misc.imread(path1).astype(np.float)
#ret4 = tiff.imread(path1).astype(np.float)
#print("shape: ", ret1.shape)
#print("shape cv2 read:", ret1.shape)
ret2 = ret1.astype(np.uint16)
print("ret1:", ret1[0:2,0:2])
print("ret2:", ret2[0:2,0:2])
#print("ret1_faltten:", ret1[0:2,0:2])

#io.imwrite('./Try/ret2.tiff', ret2)
scipy.misc.imsave("./Try/ret1.png", ret1)
#cv2.imwrite("./Try/cv2.tiff", ret1)
#tiff.imsave('./Try/tif.tiff', ret4)

retn = scipy.misc.imread("./Try/ret1.png")
print("retn:", retn[0:2,0:2])