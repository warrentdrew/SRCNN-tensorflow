import numpy
import math

# PSNR
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 65535.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))