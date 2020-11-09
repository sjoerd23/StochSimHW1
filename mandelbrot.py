import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm


class Element:
    """
    A class used to represent the number of iterations and current
    value of a single element of the mandelbrot set
    """
    def __init__(self, iteration, value):
        self.iteration = iteration
        self.value = value


class Mandelbrot:
    """
    A class used to represent and calculate Mandelbrot sets

    ...

    Attributes
    ----------
    """
    def __init__(self, dims, grid_size, mandel_max_iter):
        """
        Parameters
        ----------
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        grid_size : int
            resolution of the search space
        mandel_max_iter : int
            maximum iterations of mandelbrot equation
        """
        self.dims = dims
        self.grid_size = grid_size
        self.mandel_max_iter = mandel_max_iter
        self.fname = "results/mandelbrot_{}_{}.npy".format(self.grid_size, self.mandel_max_iter)

        # create grid points to evaluate for mandelbrot set
        self.x_grid = np.linspace(dims[0], dims[1], self.grid_size)
        self.y_grid = np.linspace(dims[2], dims[3], self.grid_size)

        # create matrix to store number of iterations to escape the set
        # if it doesnt escape: iterations is set to mandel_max_iter
        self.mandelbrot_set = np.zeros((self.grid_size, self.grid_size))

    def calc_mandelbrot_set(self, mandel_max_iter=None):
        """Create mandelbrot set

        Args:
            mandel_max_iter : int
                maximum number of iterations per candidate number c

        Returns:
            mandelbrot_set : 2D numpy array
                matrix of number of iterations before escaping mandelbrot
        """
        if mandel_max_iter is None:
            mandel_max_iter = self.mandel_max_iter

        time_start = time.time()
        for i, x in enumerate(tqdm.tqdm(self.x_grid)):
            for j, y in enumerate(self.y_grid):
                c = complex(x, y)
                z = 0
                k = 0
                while abs(z) <= 2 and k < mandel_max_iter:

                    # mandelbrot formula z_n+1 = (z_n)^2 + c
                    z = z*z + c
                    k += 1

                # save the number of iterations
                self.mandelbrot_set[j][i] = k - 1

        print("Time to calculate mandelbrot set: {:.2f} s".format(time.time() - time_start))

        return self.mandelbrot_set

    def plot_mandelbrot(self, mandelbrot_set=None, save_figure=False, save_true_size=False):
        """Plot mandelbrot set using plt.imshow() based on a matrix filled with the number of
        iterations. If you have a matrix filled with Elements, first use
        self.mandelbrot_elem_to_iter() on the mandelbrot set.
        """
        if mandelbrot_set is None:
            mandelbrot_set = self.mandelbrot_set
        fig, ax = plot_layout()
        plt.title("Mandelbrot set")
        plt.imshow(mandelbrot_set, extent=self.dims)
        if save_figure:
            plt.savefig("results/mandelbrot_{}_{}.png".format(self.grid_size, self.mandel_max_iter), dpi=1000)
        if save_true_size:
            plt.imsave("results/mandelbrot_{}_{}_true_size.png".format(self.grid_size, self.mandel_max_iter), \
                arr=mandelbrot_set, format='png')

        return

    def load_mandelbrot(self):
        """Load mandelbrot set """
        self.mandelbrot_set = np.load(self.fname)

    def save_mandelbot(self):
        """Save mandelbrot set """
        np.save(self.fname, self.mandelbrot_set)


# DISCLAIMER: This implementation should be completely correct
def latin_hypercube_sampling(dims, n_samples, antithetic):
    """Latin hypercube sampling (LHS)

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples : int
            number of random numbers to throw
        antithetic : boolean
            use antithetic variate

    Returns:
        x_rands : list
            list of randomly sampled x values using LHS
        x_rands : list
            list of randomly sampled y values using LHS
    """
    # not sure if using antithetic variables makes sense here
    antithetic = False

    x_rands = []
    y_rands = []

    # how many rows and cols do we want? I think n_rows_cols = n_samples
    n_rows_cols = int(np.sqrt(n_samples))
    n_rows_cols = n_samples

    interval_x = dims[1] - dims[0]
    interval_y = dims[3] - dims[2]
    square_width = int(interval_x / n_rows_cols)
    square_length = int(interval_y / n_rows_cols)

    for i in range(n_rows_cols):
        row_coord = i * interval_x / n_rows_cols + dims[0]
        col_coord = i * interval_y / n_rows_cols + dims[2]

        # throw a random number in the space spanned by row_coord, col_coord
        x_rand = np.random.random()
        y_rand = np.random.random()

        x_rands.append(row_coord + x_rand * square_width)
        y_rands.append(col_coord + y_rand * square_length)

        # not sure if using antithetic variables makes sense here
        # that is why antithetic is set to False at the start of this function
        if antithetic:
            x_rand_antithetic = 1 - x_rand
            y_rand_antithetic = 1 - y_rand
            x_rands.append(row_coord + x_rand_antithetic * square_width)
            y_rands.append(col_coord + y_rand_antithetic * square_length)

    # randomly permutes both x and y arrays to make random pairs
    np.random.shuffle(x_rands)
    np.random.shuffle(y_rands)

    return x_rands, y_rands

def orthogonal_sampling(dims, n_samples):
    """Orthogonal Sampling (OS)

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples : int
            number of random numbers to throw

    Returns:
        x_rands : list
            list of randomly sampled x values using OS
        x_rands : list
            list of randomly sampled y values using OS
    """
    if np.sqrt(n_samples) % 1 != 0:
        raise ValueError("The number of samples needs to be the square of an integer")

    # initialise the grids and determine scale factor
    major = int(np.sqrt(n_samples))
    xlist = np.zeros((major, major))
    ylist = np.zeros((major, major))
    xscale = (dims[1] - dims[0]) / n_samples
    yscale = (dims[3] - dims[2]) / n_samples


    m = 0
    for i in range(major):
        for j in range(major):
            xlist[i][j] = m
            ylist[i][j] = m
            m += 1


    x_rands = []
    y_rands = []

    # permute the major columns and rows
    for i in range(major):
        np.random.shuffle(xlist[i])
        np.random.shuffle(ylist[i])

    # assign a random value to each minor grid
    for i in range(major):
        for j in range(major):
            x = dims[0] + xscale * (xlist[i][j] + np.random.random())
            y = dims[2] + yscale * (ylist[j][i] + np.random.random())
            x_rands.append(x)
            y_rands.append(y)

    return x_rands, y_rands

def pure_random_sampling(dims, n_samples, antithetic):
    """Pure random sampling (PRS)

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples : int
            number of random numbers to throw
        antithetic : boolean
            use antithetic variate

    Returns:
        x_rands : list
            list of randomly sampled x values using PRS
        x_rands : list
            list of randomly sampled y values using PRS
    """

    x_rands = []
    y_rands = []

    if antithetic:
        n_samples = int(n_samples / 2)
    for i in range(n_samples):

        # throw a random number in the space spanned by dims
        x_rand = np.random.random()
        y_rand = np.random.random()
        x_rands.append((dims[1]-dims[0]) * x_rand + dims[0])
        y_rands.append((dims[3]-dims[2]) * y_rand + dims[2])
        if antithetic:
            x_rand_antithetic = 1 - x_rand
            y_rand_antithetic = 1 - y_rand
            x_rands.append((dims[1]-dims[0]) * x_rand_antithetic + dims[0])
            y_rands.append((dims[3]-dims[2]) * y_rand_antithetic + dims[2])

    return x_rands, y_rands

def plot_layout():
    """Standard plot layout for figures """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#e8e8e8', axisbelow=True)

    return fig, ax

##### NEWLY USED
def integrate_mandelbrot(dims, n_samples, n_iterations, sampling="PRS", antithetic=True):
    """Integrate the mandelbrot set using Monte Carlo method

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples : int
            number of points to throw and evaluate
        n_iterations : int
            number of iterations per candidate number c
        sampling : string ("PRS", "LHS" or "OS", default: "PRS")
            sampling technique to use for generating random numbers
        antithetic : boolean
            use antithetic variate

    Returns:
        area : float
            estimated area of integral
    """
    if sampling not in ["PRS", "LHS", "OS"]:
        print("Selected non-valid sampling technique. Switched to default: PRS")
        sampling = "PRS"

    # get n_samples random variables to throw as samples
    if sampling == "PRS":
        x_rands, y_rands = pure_random_sampling(dims, n_samples, antithetic)
    elif sampling == "LHS":
        x_rands, y_rands = latin_hypercube_sampling(dims, n_samples, antithetic)
    elif sampling == "OS":
        x_rands, y_rands = orthogonal_sampling(dims, n_samples)
    hit = 0
    miss = 0

    # for the amount of n_samples, throw a sample and evaluate if it falls in the mandelbrot in
    # n_iterations of the mandelbrot equation
    for i in tqdm.tqdm(range(len(x_rands))):
        x_rand = x_rands[i]
        y_rand = y_rands[i]
        if point_in_mandelbrot(complex(x_rand, y_rand), n_iterations):
            hit += 1
        else:
            miss += 1

    area = (dims[1] - dims[0]) * (dims[3] - dims[2])

    # return area of mandelbrot set
    return hit / (miss + hit) * (area)

def point_in_mandelbrot(c, n_iterations):
    z = 0
    for j in range(n_iterations):

        # mandelbrot formula z_n+1 = (z_n)^2 + c
        z = z*z + c
        if abs(z) > 2:

            # point is not in mandelbrot set
            return False
    else:
        # point is inside mandelbrot set
        return True

def convergence_mandelbrot(dims, n_samples, max_n_iterationsantithetic):
    """Calculate the convergence rate of the integral of the mandelbrot set depending on
    the number of iterations for the evaluations of points in the mandelbrot set

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples : int
            number of points to throw and evaluate
        max_n_iterations : int
            maximum number of iterations per candidate number c
        sampling : string ("PRS", "LHS" or "OS", default: "PRS")
            sampling technique to use for generating random numbers
        antithetic : boolean
            use antithetic variate
    """

    return

def main():

    np.random.seed()

    antithetic = True               # use antithetic variables while integrating
    dims = [-2, 0.6, -1.1, 1.1]     # dimensions of the search space [x1, x2, y1, y2]
    grid_size = 1000                # amount of grid points in each dimension
    mandel_max_iter = 256           # maximum of iterations for plotting of mandelbrot set

    ###############################################################################################
    ## this part is for creating and plotting a mandelbrot set using a grid
    ###############################################################################################
    # set to True if you want to calculate a new mandelbrot set
    # set to False if you want to load an existing mandelbrot set from fname
    # calc_new_mandelbrot_set = True

    # # create a new mandelbrot object
    # mandelbrot = Mandelbrot(dims, grid_size, mandel_max_iter)
    # if calc_new_mandelbrot_set == True:
    #     mandelbrot.calc_mandelbrot_set()
    #     mandelbrot.save_mandelbot()
    # else:
    #     mandelbrot.load_mandelbrot()

    # mandelbrot.plot_mandelbrot(save_figure=False, save_true_size=False)

    ###############################################################################################
    ## integrate mandelbrot set
    ###############################################################################################
    n_samples = 100000
    n_iterations = 256
    time_start = time.time()
    mandelbrot_area = integrate_mandelbrot(dims, n_samples, n_iterations, sampling="OS", antithetic=antithetic)
    print("Time to calculate integral: {:.2f} s".format(time.time() - time_start))
    print("The integral of the mandelbrot set is {:.6f}".format(mandelbrot_area))

    # show all plots
    plt.show()

    return


# execute file
if __name__ == "__main__":
    main()
