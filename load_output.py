import numpy as np

file = np.load(r'D:\Everest\input1_fengwu.npy')
file1 = np.load('input1.npy')
#print(file)
print(file.shape)
print('np.nanmax(file):', np.nanmax(file))

print(file1.shape)
print('np.nanmax(file1):', np.nanmax(file1))