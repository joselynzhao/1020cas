#-*-coding:utf-8-*-

import numpy as np
import PIL.Image as Image

info = np.load('info.npy')
info_1 = info[3,0]
info_1 *= 255
info_1 = np.array(info_1,dtype=np.uint8)
# info_1 = info_1.transpo//////////////////////////////////////////////////////////////////////////////////////////////se(3,1,2)
# info  = info_1.resize(256,256,3)
img = Image.fromarray(info_1)
img.show()
print(info_1.max())
print(info_1.min())

pass
