import numpy as np

a = np.array([1, 0, 0, 0, 0]).astype("int64")
b = np.array([[1, 1, 1, 1, 1],]).astype("int32")
b = b.reshape(5, 1)
print(a.shape)
print(b.shape)
print(np.mean(a == b))
