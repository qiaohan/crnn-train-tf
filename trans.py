import numpy as np
import os

np.fromfile("bin/logit_weights.bin",np.float32).reshape([2048,3851]).transpose().tofile("bins/logit_weights.bin")
shapes = [
        (3, 3, 3, 64),
        (3, 3, 64, 128),
        (3, 3, 128, 256),
        (3, 3, 256, 256),
        (3, 3, 256, 512),
        (3, 3, 512, 512),
        (3, 3, 512, 512)
        ]
froot = "bin/"
for i in range(1,8):
    f = "conv"+str(i)+"_weights.bin"
    tt = np.fromfile(froot+f,np.float32)
    tt.reshape(shapes[i-1]).transpose([3,2,0,1]).tofile("bins/"+f)
