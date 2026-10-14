import numpy as np

b=np.load('output.npy')
print(b.shape)
print(np.nanmax(b))