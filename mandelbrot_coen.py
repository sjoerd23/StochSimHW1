import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cmath 
import time
t = time.time()

def comp_it(c, max_iterations):
	"""
	computes the number of iterations the absolute value of n is below 2
	capped at max_iterations
	"""
	n = 0
	j = 0
	while abs(n) <= 2 and j <= max_iterations: 
		n = n**2 + c
		j += 1
	return j - 1

def mandelbrot(dim, size, it):
	""" Evaluate the mandelbrot set

	Parameters
	----------
	dim : list [x1, x2, y1, y2]
		dimension of the search space
	size : int 
		resolution of the search space
	it : int
		maximum number of iterations

	Returns 
	-------
	matrix
		a matrix containing the number of iterations within the mandelbrot equation
	"""	

	x_dim = [dim[0], dim[1]]
	y_dim = [dim[2], dim[3]]

	x_set = np.linspace(dim[0], dim[1], size)
	y_set = np.linspace(dim[2], dim[3], size)
	m = np.zeros((size, size))

	for i, x in enumerate(x_set):
		for j, y in enumerate(y_set):
			c = complex(x, y)
			m[j][i] = comp_it(c, iterations)
	return m

dimensions = [-2, 0.5, -1.3, 1.3]
surface = (dimensions[1] - dimensions[0]) * (dimensions[3] - dimensions[2])
size = 1000
iterations = 256

m = mandelbrot(dimensions, size, iterations)

hit = 0
miss = 0
for i in range(10000):
	randx = np.random.randint(size)
	randy = np.random.randint(size)
	if m[randy][randx] == iterations:
		hit += 1
	else:
		miss += 1

area_mandelbrot = hit / (miss + hit) * (surface)
print("The area is {}".format(area_mandelbrot))

t2 = time.time()
print(t2 - t)
plt.figure()
plt.imshow(m, extent=dimensions)
plt.show()