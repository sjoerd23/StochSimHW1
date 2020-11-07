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

    def calc_conv_mandelbrot(self, mandel_max_iter=None):
        """Calculate the convergence rate of the integral of the mandelbrot set depending on
        the number of iterations for the creation of the mandelbrot set """
        if mandel_max_iter is None:
            mandel_max_iter = self.mandel_max_iter

        mandelbrot_set_conv = np.zeros((self.grid_size, self.grid_size), dtype=object)

        # initiate Element for each grid point
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                mandelbrot_set_conv[j][i] = Element(0, 0)

        # store all calculated areas
        areas_conv = []

        # calculate all iterations and for each iterations the integral
        for mandel_iter in range(mandel_max_iter):
            print("Progress: {:.2%}".format(mandel_iter/mandel_max_iter))
            mandelbrot_set_conv = self.single_iteration(mandelbrot_set_conv, mandel_iter)

            # to integrate we need a mandelbrot_set without Elements objects inside, but just the
            # amount of iterations
            mandelbrot_set_conv_iter = self.mandelbrot_elem_to_iter(mandelbrot_set_conv)
            areas_conv.append(self.integrate_mandelbrot(mandelbrot_set=mandelbrot_set_conv_iter, mandel_max_iter=mandel_iter+2))

        # calculate A_js - A_is
        area_diffs = [abs(area_conv - areas_conv[-1]) for area_conv in areas_conv]
        max_iters = [max_iter for max_iter in range(mandel_max_iter)]

        # plot convergence rate of integral value
        fig2, ax2 = plot_layout()
        plt.title("Absolute difference in area over number of iterations")
        plt.plot(max_iters, area_diffs)
        plt.xlabel("Iterations")
        plt.ylabel("Differences in area")
        # plt.savefig("results/mandelbrot_diffs_iter_{}_{}.png".format(grid_size, mandel_max_iter), dpi=1000)
        return

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

    def mandelbrot_elem_to_iter(self, mandelbrot_set_elem):
        """Convert mandelbrot set consisting of Elements to mandelbrot set consisting of Iterations
        """
        mandelbrot_set_iter = np.zeros((self.grid_size, self.grid_size), dtype=np.int)

        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                mandelbrot_set_iter[j][i] = int(mandelbrot_set_elem[j][i].iteration)

        return mandelbrot_set_iter

    def integrate_mandelbrot_stat(self, n_repetitions=100, mandelbrot_set=None, mandel_max_iter=None, sampling="PRS", antithetic=True, mc_max_iter=100000):
        """Integrate the mandelbrot set n_repitions times using self.integrate_mandelbrot()

        Args:
            n_repetitions : int
                the amount of integrals evaluated
            mandelbrot_set : numpy 2D array
                mandelbrot set, if None defaults to self.mandelbrot_set
            mandel_max_iter : int
                maximum number of iterations per candidate number c
            sampling : string ("PRS" or "LHS", default: "PRS")
                sampling technique to use for generating random numbers
            antithetic : boolean
                use antithetic variate
            mc_max_iter : int
                number of random points to throw for integration

        Returns:
            areas : list of floats
                estimated areas of integral
            mean : float
                mean of areas
            sample_stddev : float
                standard deviation of areas
            var : float
                variance of areas
            interval : float
                confidence interval of p = 95%
        """
        areas = np.zeros(n_repetitions)
        for i in range(n_repetitions):
            areas[i] = self.integrate_mandelbrot(mandelbrot_set=mandelbrot_set, mandel_max_iter=mandel_max_iter, sampling=sampling)

        mean = np.mean(areas)
        sample_stddev = np.std(areas, ddof=1) # we want sample stddev, so ddof=1
        var = np.var(areas)

        # calculate confidence interval with p = 95% -> lambda 1.96
        # a = lambda sample_stddev / sqrt(n_repetitions)
        lambda_p = 1.96
        interval = lambda_p * sample_stddev / np.sqrt(n_repetitions)
        return areas, mean, sample_stddev, var, interval

    def integrate_mandelbrot(self, mandelbrot_set=None, mandel_max_iter=None, sampling="PRS", antithetic=True, mc_max_iter=100000):
        """Integrate the mandelbrot set using Monte Carlo method. User can set antithetic to True
        to use antithetic variables

        Args:
            mandelbrot_set : numpy 2D array
                mandelbrot set, if None uses self.mandelbrot_set
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
        if mandelbrot_set is None:
            mandelbrot_set = self.mandelbrot_set
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

            # preferably grid_size is dividable by partition_size
            partition_size = 5
            n_partitions = int(self.grid_size/partition_size)
            x_rands, y_rands = latin_hypercube_sampling(self.grid_size, n_partitions)
        else:
            # something went wrong, terminate program
            print("Non-valid sampling technique. Terminating program...")
            exit(1)

        # for each coordinate in (x_rands, y_rands) throw it on the grid and evaluate it as miss or hit
        for i in range(len(x_rands)):

            x_rand = x_rands[i]
            y_rand = y_rands[i]
            if mandelbrot_set[y_rand][x_rand] == mandel_max_iter - 1:
                hit += 1
            else:
                miss += 1

            # get extra random number to evaluate when using antithetic method:
            if antithetic:
                x_rand_antithetic = 1 - x_rand
                y_rand_antithetic = 1 - y_rand

                if mandelbrot_set[y_rand_antithetic][x_rand_antithetic] == mandel_max_iter - 1:
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

        return

    def single_iteration(self, mandelbrot_set, mandel_iter):
        """Computes a single iterations of the equation
        n_2 = n_1**2 + c for all elements within abs(n) < 2
        """
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                element = mandelbrot_set[j][i]
                value = element.value
                if element.iteration == mandel_iter:
                    c = complex(x, y)
                    element.value = value * value + c
                    if abs(element.value) <= 2:
                        element.iteration = mandel_iter + 1
                        # mandelbrot_set[j][i] = Element(mandel_iter+1, next_value)

        return mandelbrot_set

    def load_mandelbrot(self):
        """Load mandelbrot set """
        self.mandelbrot_set = np.load(self.fname)

    def save_mandelbot(self):
        """Save mandelbrot set """
        np.save(self.fname, self.mandelbrot_set)


# DISCLAIMER: This implementation should be completely correct
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
    grid_size = 1000                # amount of grid points in each dimension
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
    mandelbrot_area = mandelbrot.integrate_mandelbrot(sampling="LHS")
    print("The integral of the mandelbrot set is {:.6f}".format(mandelbrot_area))

    # calculate areas, mean, sample_stddev, var and confidence interval for n_repetitions times of integrating mandelbrot set
    areas, mean, sample_stddev, var, interval = mandelbrot.integrate_mandelbrot_stat(n_repetitions=10, sampling="LHS")
    print("Mean: {}".format(mean))
    print("Standard deviation: {}".format(sample_stddev))
    print("Variance: {}".format(var))
    print("95% confidence interval: {}".format(interval))

    # investigate the convergence with mandel_max_iter
    # mandelbrot.calc_conv_mandelbrot()

    # show all plots
    plt.show()

    return


# execute file
if __name__ == "__main__":
    main()
