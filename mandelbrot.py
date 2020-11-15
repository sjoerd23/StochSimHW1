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
            plt.savefig(
                "results/mandelbrot_{}_{}.png".format(self.grid_size, self.mandel_max_iter),
                dpi=1000
            )
        if save_true_size:
            plt.imsave(
                "results/mandelbrot_{}_{}_true_size.png"
                .format(self.grid_size, self.mandel_max_iter),
                arr=mandelbrot_set, format='png'
            )

        return

    def load_mandelbrot(self):
        """Load mandelbrot set """
        self.mandelbrot_set = np.load(self.fname)

    def save_mandelbot(self):
        """Save mandelbrot set """
        np.save(self.fname, self.mandelbrot_set)


# DISCLAIMER: This implementation should be completely correct
def latin_hypercube_sampling(dims, n_samples, antithetic=False):
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


def pure_random_sampling(dims, n_samples, antithetic=False):
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


def convergence_mandelbrot(dims, n_samples_all, max_n_iterations, sampling, antithetic, runs=1):
    """Calculate the convergence rate of the integral of the mandelbrot set depending on
    the number of iterations for the evaluations of points in the mandelbrot set

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples_all : list
            number of points to throw and evaluate
        max_n_iterations : int
            maximum number of iterations per candidate number c
        sampling : string ("PRS", "LHS" or "OS", default: "PRS")
            sampling technique to use for generating random numbers
        antithetic : boolean
            use antithetic variate
        runs : integer
            number of time the same evaluation is done
    """

    fig, ax = plot_layout()
    plt.title("Absolute difference in area over number of iterations")

    offset = 0

    for n_samples in n_samples_all:
        area_diffs_all = np.zeros((runs, max_n_iterations - offset))
        for j in range(runs):
            areas = np.zeros(max_n_iterations - offset)
            iters = np.zeros(max_n_iterations - offset)
            for i in tqdm.tqdm(range(offset, max_n_iterations)):
                areas[i-offset] = integrate_mandelbrot(dims, n_samples, i+1, sampling=sampling)
                iters[i-offset] = i + 1

            # calculate A_js - A_is
            area_diffs = [abs(area - areas[-1]) for area in areas]
            area_diffs_all[j] = area_diffs

        mean_area_diff = np.array([np.mean(area_diffs_all[:, x])
                                   for x in range(max_n_iterations - offset)])
        std_area_diff = np.array([np.std(area_diffs_all[:, x], ddof=1)
                                  for x in range(max_n_iterations - offset)])

        # plot convergence rate of integral value, to plot with line, change fmt to fmt=""
        plt.errorbar(
            iters, mean_area_diff, marker=".", fmt=".", solid_capstyle="projecting", capsize=5,
            yerr=std_area_diff, label="samples drawn; {}".format(n_samples)
        )

    plt.legend()
    plt.xlabel("Iterations [-]")
    plt.ylabel("Differences in area [-]")
    plt.savefig(
        "results/mandelbrot_diffs_iter_{}_{}.png".format(n_samples, max_n_iterations), dpi=1000
    )

    return mean_area_diff, std_area_diff


def conf_int_mandelbrot(dims, n_samples_all, n_iterations, sampling_all, antithetic, runs=1):
    """Calculate the confidence interval for the integral of the mandelbrot set depending on
    the number of the samples drawn

    Args:
        dims : list [x1, x2, y1, y2]
            dimension of the search space
        n_samples_all : list
            number of points to throw and evaluate
        n_iterations : int
            maximum number of iterations per candidate number c
        sampling : list ("PRS", "LHS" or "OS", default: "PRS")
            sampling technique to use for generating random numbers
        antithetic : boolean
            use antithetic variate
        runs : integer
            number of time the same evaluation is done
    """

    fig, ax = plot_layout()
    ax.set_title("Confidence interval for the integral of the area")
    fig2, ax2 = plot_layout()

    areas = np.zeros((len(n_samples_all), runs))

    mean_area = np.zeros((len(sampling_all), len(n_samples_all)))
    conf_area = np.zeros((len(sampling_all), len(n_samples_all)))
    for k, sampling in enumerate(sampling_all):
        for i, n_samples in enumerate(n_samples_all):
            for j in range(runs):
                areas[i][j] = integrate_mandelbrot(dims, n_samples, n_iterations, sampling=sampling)

        mean_area[k] = [np.mean(areas[x]) for x in range(len(n_samples_all))]
        conf_area[k] = [(np.std(areas[x], ddof=1) * 1.96) / np.sqrt(runs)
                        for x in range(len(n_samples_all))]

        # plot convergence rate of integral value
        ax.scatter(n_samples_all, conf_area[k], label="Sampling: {}".format(sampling))
        ax2.scatter(n_samples_all, mean_area[k], label="Sampling: {}".format(sampling))

    ###### remove in final version
    ## Give appropriate filename yourself
    custom_fname = "sim_0"
    fname = "results/" + custom_fname
    np.savetxt(fname+"areas.txt", areas)
    np.savetxt(fname+"mean_area.txt", mean_area)
    np.savetxt(fname+"conf_area.txt", conf_area)

    ax.set_xlabel("Samples drawn [-]")
    ax.set_ylabel("Confidence interval [-]")
    ax2.set_xlabel("Samples drawn [-]")
    ax2.set_ylabel("Area [-]")
    ax.legend()
    ax2.legend()
    # plt.savefig(
    #     "results/mandelbrot_diffs_iter_{}_{}.png".format(n_samples, max_n_iterations), dpi=1000
    # )

    return mean_area, conf_area


def main():

    np.random.seed()

    antithetic = False               # use antithetic variables while integrating
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
    # n_samples = 1000**2
    # n_iterations = 256

    # # perform integration for all sampling techniques
    # samplings = ["PRS", "LHS", "OS"]
    # for sampling in samplings:
    #     time_start = time.time()
    #     mandelbrot_area = integrate_mandelbrot(
    #         dims, n_samples, n_iterations, sampling=sampling, antithetic=antithetic
    #     )
    #     print(
    #         "Time to calculate integral with {}: {:.2f} s"
    #         .format(sampling, time.time() - time_start)
    #     )
    #     print(
    #         "The integral of the mandelbrot set with {} is {:.6f}\n"
    #         .format(sampling, mandelbrot_area)
    #     )

    ###############################################################################################
    ## investigate confidence interval for different values of n_samples
    ###############################################################################################

    # n_samples = [i ** 2 for i in range(200, 600, 20)]
    # n_samples = [200**2, 250**2, 300**2, 350**2,400**2]
    # n_samples = [200**2]
    # max_n_iterations = 256
    # sampling = ["PRS"]
    # m, c = conf_int_mandelbrot(
    #     dims, n_samples, max_n_iterations, sampling_all=sampling, antithetic=antithetic, runs = 5
    # )
    # for i in range(len(m)):
    #     print(
    #         "The area of the mandelbrot set is estimated at {:.5f} +- {:.5f} for {} sampling"
    #         .format(m[i][-1], c[i][-1], sampling[i])
    #     )

    ###############################################################################################
    ## investigate confidence interval for different values of n_samples for different sampling techniques
    ###############################################################################################

    # Confidence interval for the different types of sampling
    # n_samples = [i ** 2 for i in range(1000, 2200, 250)]
    # n_samples = [512**2, 1024**2, 2048**2, 4069**2]
    # n_samples = [1250**2]
    # max_n_iterations = 4096*2
    #
    # sampling = ["PRS"]
    # m, c = conf_int_mandelbrot(
    #     dims, n_samples, max_n_iterations, sampling_all=sampling, antithetic=antithetic, runs=5
    # )
    # for i in range(len(m)):
    #     print(
    #         "The area of the mandelbrot set is estimated at {:.5f} +- {:.5f} for {} sampling"
    #         .format(m[i][-1], c[i][-1], sampling[i])
    #     )

    ###############################################################################################
    ## investigate the convergence rate over n_iterations with fixed n_samples
    ###############################################################################################
    time_start = time.time()
    n_samples = [100**2, 200**2, 300**2]
    max_n_iterations = 25
    mean, std = convergence_mandelbrot(
        dims, n_samples, max_n_iterations, sampling="PRS", antithetic=antithetic, runs = 3
    )
    print("Time to calculate convergence rate: {:.2f} s".format(time.time() - time_start))

    # look at the samplings
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # axes = axes.flatten()
    # n_samples = 5**2
    # samplings = [pure_random_sampling, latin_hypercube_sampling, orthogonal_sampling]
    # for i, sampling in enumerate(samplings):
    #     x_rands, y_rands = sampling(dims, n_samples)
    #
    #     axes[i].scatter(x_rands, y_rands)
    #     axes[i].set_title("{}".format(sampling.__name__))

####################################################################################
    # Load simulation (hacky code, remove before handing in)
####################################################################################

    # set this by hand
    # sampling_all = ["PRS"]
    # n_samples_all = [50**2, 60**2]
    # custom_fname = "sim_0"
    #
    # fname = "results/" + custom_fname
    # areas = np.array(np.loadtxt(fname+"areas.txt"))
    # mean_area = np.array(np.loadtxt(fname+"mean_area.txt"))
    # conf_area = np.array(np.loadtxt(fname+"conf_area.txt"))
    # print(areas)
    # print(conf_area, mean_area)
    # if len(n_samples_all) == 1 and len(sampling_all) == 1:
    #     mean_area = np.array([np.loadtxt(fname+"mean_area.txt")]).flatten()
    #     conf_area = np.array([np.loadtxt(fname+"conf_area.txt")]).flatten()
    #
    # fig, ax = plot_layout()
    # ax.set_title("Confidence interval for the integral of the area")
    # fig2, ax2 = plot_layout()
    # ax2.set_title("Integral value")
    #
    # for k, sampling in enumerate(sampling_all):
    #
    #     if len(sampling_all) == 1 and len(n_samples_all) > 1:
    #         conf_area_k = conf_area
    #         mean_area_k = mean_area
    #     else:
    #         conf_area_k = conf_area[k]
    #         mean_area_k = mean_area[k]
    #     # plot convergence rate of integral value
    #     ax.scatter(n_samples_all, conf_area_k, label='Sampling: {}'.format(sampling))
    #     ax2.scatter(n_samples_all, mean_area_k, label='Sampling: {}'.format(sampling))
    #
    # ###### remove in final version
    # ## Give appropriate filename yourself
    #
    # ax.set_xlabel("Samples drawn [-]")
    # ax.set_ylabel("Confidence interval [-]")
    # ax2.set_xlabel("Samples drawn [-]")
    # ax2.set_ylabel("Area [-]")
    # ax.legend()
    # ax2.legend()

    # show all plots
    plt.show()

    return


# execute file
if __name__ == "__main__":
    main()
