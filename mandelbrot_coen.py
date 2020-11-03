import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cmath 
import time
from tqdm import tqdm
t = time.time()

# This class is solely created to be able to include both the iterations
# AND the current value of the element in the np.array
class Element:
	"""
	A class used to represent the number of iterations and current 
	value of a single element of the mandelbrot set
	"""
	def __init__(self, it, value):
		self.it = it
		self.value = value


class Mandelbrot_set:
	"""
	A class used to represent a mandelbrot set
	
	...

	Attributes
	----------
	dim : list [x1, x2, y1, y2]
		dimension of the search space
	size : int 
		resolution of the search space

	Methods
	-------
	single_it()
		Computes a single iterations of the equation
		n_2 = n_1**2 + c for all elements within abs(n) < 2
	"""
	def __init__(self, dim, size):
		"""
		Parameters
		----------
		dim : list [x1, x2, y1, y2]
			dimension of the search space
		size : int 
			resolution of the search space
		"""
		self.dim = dim
		self.size = size
		self.it = 0
		self.x_set = np.linspace(dim[0], dim[1], size)
		self.y_set = np.linspace(dim[2], dim[3], size)
		self.set = np.zeros((size, size),dtype=object)
		for i, x in enumerate(self.x_set):
			for j, y in enumerate(self.y_set):
				self.set[j][i] = Element(0, 0)


	def single_it(self):
		"""
		compute one single iteration of the mandelbrot equation for all
		elements
		"""
		evals = 0
		for i, x in enumerate(self.x_set):
			for j, y in enumerate(self.y_set):
				element = self.set[j][i]
				iterations = element.it
				value = element.value
				if iterations == self.it: 
					evals += 1
					c = complex(x, y)
					next_val = value**2 + c
					if abs(next_val) < 2:
						self.set[j][i] = Element(self.it + 1, next_val)

		self.it += 1


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

	for i, x in enumerate(tqdm(x_set)):
		for j, y in enumerate(y_set):
			c = complex(x, y)
			m[j][i] = comp_it(c, it)
	return m

def monte_carlo(m, N, size, surface):
	hit = 0
	miss = 0
	for i in range(N):
		randx = np.random.randint(size)
		randy = np.random.randint(size)
		if m[randy][randx] == iterations:
			hit += 1
		else:
			miss += 1

	return hit / (miss + hit) * (surface)

def save_data(m, fname):
	np.save(fname, m)

def load_data(fname):
	return np.load(fname)

def plot(m):
	plt.figure()
	plt.imshow(m, extent=dimensions, cmap="gist_earth")
	plt.show()


if __name__ == "__main__":
	# parameters
	dimensions = [-2, 0.5, -1.3, 1.3]
	surface = (dimensions[1] - dimensions[0]) * (dimensions[3] - dimensions[2])
	size = 1000
	iterations = 400
	fname = "s{},i{},x1{},x2{},y{}.npy".format(size, iterations, dimensions[0], dimensions[1], dimensions[3])
	N = 10000

	t1 = time.time()
	m = Mandelbrot_set(dimensions, size)

	for i in range(80):
		print(i)
		m.single_it()


	t3 = time.time()
	print("time for new method: {}".format(t3-t1))

	for i in range(80):
		# t2 = time.time()
		p = mandelbrot(dimensions, size, i)
		# print(time.time() - t2)

	print("time for old method: {}".format(time.time() - t3))

	# m = mandelbrot(dimensions, size, iterations)
	# save_data(m, fname)

	# m = load_data(fname)
	# plot(m)
	# area = monte_carlo(m, N, size, surface)
	# print(area)


	# t2 = time.time()
	# print(t2 - t)
