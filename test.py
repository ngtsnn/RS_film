import numpy as np

a = np.array([2, 4, 9, 8, 8, 4, 2, 5, 5, 4, 20])
b = np.array([2, 4, 5, 8, 9])
c = np.random.rand(10) * 100

# z = np.zeros(shape=(10, 2))
# z[:, 0] = a
# z[:, 1] = c

# for l in b:
#   indices = np.where(a == l)
#   z[indices, 0] = np.where(b == l)
s = np.where(a == b[0])

print(s)