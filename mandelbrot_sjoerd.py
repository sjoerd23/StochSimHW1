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

    def integrate_mandelbrot(self, mandel_max_iter=None, sampling="PRS", antithetic=True, mc_max_iter=100000):
        """Integrate the mandelbrot set using Monte Carlo method. User can set antithetic to True
        to use antithetic variables

        Args:
            mandel_max_iter : int
                maximum number of iterations per candidate number c
            sampling : string ("PRS" or "LHS", default: "PRS")
                sampling technique to use for generating random numbers
            antithetic : boolean
                use antithetic variate
            mc_max_iter : int
                number of random points to throw for integration

        Returns:
            area : float
                estimated area of integral
        """
        if mandel_max_iter is None:
            mandel_max_iter = self.mandel_max_iter

        if sampling not in ["PRS", "LHS"]:
            print("Selected non-valid sampling technique. Switched to default: PRS")
            sampling = "PRS"

        hit = 0
        miss = 0

        # if using antithetic, reduce the number of evals with a factor 2
        if antithetic:
            mc_max_iter = int(mc_max_iter / 2)

        if sampling == "PRS":
            # generate pure random sampled numbers
            x_rands, y_rands = pure_random_sampling(self.grid_size, mc_max_iter)
        elif sampling == "LHS":
            x_rands, y_rands = latin_hypercube_sampling(self.grid_size, int(self.grid_size/10))
        else:
            # something went wrong, terminate program
            print("Non-valid sampling technique. Terminating program...")
            exit(1)

        # for each coordinate in (x_rands, y_rands) throw it on the grid and evaluate it as miss or hit
        for i in range(len(x_rands)):

            x_rand = x_rands[i]
            y_rand = y_rands[i]
            if self.mandelbrot_set[y_rand][x_rand] == mandel_max_iter - 1:
                hit += 1
            else:
                miss += 1

            # get extra random number to evaluate when using antithetic method:
            if antithetic:
                x_rand_antithetic = 1 - x_rand
                y_rand_antithetic = 1 - y_rand

                if self.mandelbrot_set[y_rand_antithetic][x_rand_antithetic] == mandel_max_iter - 1:
                    hit += 1
                else:
                    miss += 1

        # calculate total area over which we integrate
        area = (self.dims[1] - self.dims[0]) * (self.dims[3] - self.dims[2])

        # return area of mandelbrot set
        return hit / (miss + hit) * (area)

    def plot_mandelbrot(self, save_figure=False, save_true_size=False):
        fig, ax = plot_layout()
        plt.title("Mandelbrot set")
        plt.imshow(self.mandelbrot_set, extent=self.dims)

        if save_figure:
            plt.savefig("results/mandelbrot_{}_{}.png".format(self.grid_size, self.mandel_max_iter), dpi=1000)
        if save_true_size:
            plt.imsave("results/mandelbrot_{}_{}_true_size.png".format(self.grid_size, self.mandel_max_iter), \
                arr=self.mandelbrot_set, format='png')
        return plt.show()

    def load_mandelbrot(self):
        """Load mandelbrot set """
        self.mandelbrot_set = np.load(self.fname)

    def save_mandelbot(self):
        """Save mandelbrot set """
        np.save(self.fname, self.mandelbrot_set)

    ### NOT WORKING AT THE MOMENT. WORK IN PROGRESS ###
    def calc_mandelbrot_all_iters(self, mandel_max_iter=None):
        """Computers mandelbrot sets for each iter in range of mandel_max_iter """
        if mandel_max_iter is None:
            mandel_max_iter = self.mandel_max_iter

        # mandelbrot_sets = np.array([np.zeros((self.grid_size, self.grid_size)) for _ in range(mandel_max_iter)])
        mandelbrot_set = np.zeros((self.grid_size, self.grid_size), dtype=object)

        # initiate Element for each grid point
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                mandelbrot_set[j][i] = Element(0, 0)

        # calculate all iteraions of mandelbrot set
        for mandel_iter in range(tqdm.tqdm(mandel_max_iter)):
            mandelbrot_set = self.single_iteration(mandelbrot_set, mandel_iter)

        return mandelbrot_set

    ### NOT WORKING AT THE MOMENT. WORK IN PROGRESS ###
    def single_iteration(self, mandelbrot_set, mandel_iter):
        """Computes a single iterations of the equation
        n_2 = n_1**2 + c for all elements within abs(n) < 2
        """
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                element = mandelbrot_set[j][i]

                if element.iteration == mandel_iter:
                    c = complex(x, y)
                    element.value = element.value*element.value + c
                    if abs(element.value) <= 2:
                        element.iteration += 1

        return mandelbrot_set

# DISCLAIMER: not sure if implementation is correct
## TODO: change to PyDOE lhs() function
def latin_hypercube_sampling(grid_size, n_partitions):
    """Latin hypercube sampling (LHS)

    Args:
        grid_size : int
            length of grid in each direction
        n_partitions : int
            number of partitions in each dimension (preferably grid_size/n_partitions=int)

    Returns:
        x_rands : list
            list of randomly sampled x values using LHS
        x_rands : list
            list of randomly sampled y values using LHS
    """
    x_rands = []
    y_rands = []

    partition_size = int(grid_size / n_partitions)
    for i in range(n_partitions):
        partition_start = int(i * grid_size / n_partitions)
        x_rands.append(partition_start + np.random.randint(partition_size))
        y_rands.append(partition_start + np.random.randint(partition_size))

    np.random.shuffle(x_rands)
    np.random.shuffle(y_rands)
    return x_rands, y_rands

def pure_random_sampling(grid_size, n_samples):
    """Pure random sampling (PRS)

    Args:
        grid_size : int
            length of grid in each direction
        n_samples : int
            number of random numbers to throw

    Returns:
        x_rands : list
            list of randomly sampled x values using PRS
        x_rands : list
            list of randomly sampled y values using PRS
    """

    x_rands = []
    y_rands = []

    for i in range(n_samples):

        # throw a random number on the grid
        x_rands.append(np.random.randint(grid_size))
        y_rands.append(np.random.randint(grid_size))

    return x_rands, y_rands

def plot_layout():
    """Standard plot layout for figures """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#e8e8e8', axisbelow=True)

    return fig, ax

def main():

    np.random.seed()

    antithetic = True               # use antithetic variables while integrating
    dims = [-2, 0.6, -1.1, 1.1]     # dimensions of the search space [x1, x2, y1, y2]
    grid_size = 500                # amount of grid points in each dimension
    mandel_max_iter = 256           # maximum of iterations for mandelbrot set

    # set to True if you want to calculate a new mandelbrot set
    # set to False if you want to load an existing mandelbrot set from fname
    calc_new_mandelbrot_set = True

    # create a new mandelbrot object
    mandelbrot = Mandelbrot(dims, grid_size, mandel_max_iter)
    if calc_new_mandelbrot_set == True:
        mandelbrot.calc_mandelbrot_set()
        mandelbrot.save_mandelbot()
    else:
        mandelbrot.load_mandelbrot()

    mandelbrot.plot_mandelbrot(save_figure=False, save_true_size=False)

    # integrate mandelbrot set
    mandelbrot_area = mandelbrot.integrate_mandelbrot()
    print("The integral of the mandelbrot set is {:.6f}".format(mandelbrot_area))



    ### NOT WORKING AT THE MOMENT. WORK IN PROGRESS ###
    ###
    # comment out next part if you only want to calculate the integral and don't want to see the
    # convergence rate (takes relatively long to calculate)
    ###
    ########################################################################################

    # investigate the convergence
    # mandelbrot_areas_conv = []
    # for max_iter in range(mandel_max_iter):
    #
    #     print("Evaluating max_iter {} of {} now...".format(max_iter, mandel_max_iter))
    #
    #     # create mandelbrot set using max_iter
    #     mandelbrot_set_conv = mandelbrot(dims, grid_size, max_iter)
    #
    #     # integrate mandelbrot set using max_iter
    #     area = integrate_mandelbrot(mandelbrot_set_conv, dims, grid_size, max_iter, sampling="PRS", antithetic=True)
    #     mandelbrot_areas_conv.append(area)
    #
    # # calculate A_js - A_is
    # area_diffs = [(mandelbrot_area_conv - mandelbrot_area) for mandelbrot_area_conv in mandelbrot_areas_conv]
    # max_iters = [max_iter for max_iter in range(mandel_max_iter)]
    #
    # # plot convergence rate of integral value
    # fig2, ax2 = plot_layout()
    # plt.title("Absolute difference in area over number of iterations")
    # plt.plot(max_iters, area_diffs)
    # plt.xlabel("Iterations")
    # plt.ylabel("Differences in area")
    # plt.savefig("results/mandelbrot_diffs_iter_{}_{}.png".format(grid_size, mandel_max_iter), dpi=1000)
    ########################################################################################

    # plt.show()
    print("Done!")

    return


# execute file
if __name__ == "__main__":
    main()
