import numpy as np
import matplotlib.pyplot as plt
import time

def integrate_mandelbrot(mandelbrot_set, dims, grid_size, mandel_max_iter, antithetic=False, mc_max_iter=10000):
    """Integrate the mandelbrot set using Monte Carlo method. User can set antithetic to True
    to use antithetic variables

    Args:
        mandelbrot_set : 2D numpy array
            matrix of number of iterations before escaping mandelbrot
        dims : list [x1, x2, y1, y2]
            dimensions of the search space
        grid_size : int
            length of grid in each direction
        mandel_max_iter : int
            maximum number of iterations per candidate number c
        antithetic : boolean
            use antithetic variate
        mc_max_iter : int
            number of random points to throw for integration

    Returns:
        surface : float
            estimated surface of integral
    """

    hit = 0
    miss = 0

    # if using antithetic, reduce the number of evals with a factor 2
    if antithetic:
        mc_max_iter = int(mc_max_iter / 2)

    for i in range(mc_max_iter):

        # throw a random number on the grid
        x_rand = np.random.randint(grid_size)
        y_rand = np.random.randint(grid_size)
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

    # calculate total surface over which we integrate
    surface = (dims[1] - dims[0]) * (dims[3] - dims[2])

    # return area of mandelbrot set
    return hit / (miss + hit) * (surface)

def mandelbrot(dims, grid_size, mandel_max_iter):
    """Create mandelbrot set

    Args:
        dims : list [x1, x2, y1, y2]
            dimensions of the search space
        grid_size : int
            length of grid in each direction
        mandel_max_iter : int
            maximum number of iterations per candidate number c

    Returns:
        mandelbrot_set : 2D numpy array
            matrix of number of iterations before escaping mandelbrot
    """

    # create grid points to evaluate for mandelbrot set
    x_grid = np.linspace(dims[0], dims[1], grid_size)
    y_grid = np.linspace(dims[2], dims[3], grid_size)

    # create matrix to store number of iterations to escape the set
    # if it doesnt escape: iterations is set to 0
    mandelbrot_set = np.zeros((grid_size, grid_size))

    # loop over all coordinates
    for i, x in enumerate(x_grid):

        # print("Progression: {:.2%}".format(i/len(x_grid)))
        for j, y in enumerate(y_grid):

            # create imaginary number c = a + bi
            c = complex(x, y)

            # max size of mandelbrot set is 2
            z = 0
            k = 0
            while abs(z) <= 2 and k < mandel_max_iter:

                # mandelbrot formula z_n+1 = (z_n)^2 + c
                z = z**2 + c
                k += 1

            # save the number of iterations. 0 denotes mandelbrot set number
            # if after mandel_max_iter z < 2 --> member of mandelbrot set
            mandelbrot_set[j][i] = k - 1

    return mandelbrot_set

def plot_layout():
    """Standard plot layout for figures """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#e8e8e8', axisbelow=True)

    return fig, ax

def load_mandelbrot(fname):
    """Load mandelbrot set """
    return np.loadtxt(fname)

def save_mandelbot(fname, mandelbrot_set):
    """Save mandelbrot set """
    return np.savetxt(fname, mandelbrot_set)

def main():

    np.random.seed()

    antithetic = True               # use antithetic variables while integrating
    dims = [-2, 0.6, -1.1, 1.1]     # dimensions of the search space [x1, x2, y1, y2]
    grid_size = 1000                # amount of grid points in each dimension
    mandel_max_iter = 256           # maximum of iterations for mandelbrot set

    fname = "results/mandelbrot_{}_{}".format(grid_size, mandel_max_iter)

    # set to True if you want to calculate a new mandelbrot set
    # set to False if you want to load an existing mandelbrot set from fname
    calculate_new_mandelbrot = True

    if calculate_new_mandelbrot == True:
        time_start = time.time()

        # calculate the mandelbrot set and save to txt file
        mandelbrot_set = mandelbrot(dims, grid_size, mandel_max_iter)
        save_mandelbot(fname, mandelbrot_set)
        print("Time to calculate mandelbrot set: {} s".format(time.time() - time_start))

    else:
        mandelbrot_set = load_mandelbrot(fname)

    # calculate integral of mandelbrot set using Monte Carlo
    ## TODO: integrate multiple times and take the mean and stddev
    mandelbrot_area = integrate_mandelbrot(mandelbrot_set, dims, grid_size, mandel_max_iter, antithetic=antithetic, mc_max_iter=100000)
    print("The integral of the mandelbrot set is {}".format(mandelbrot_area))

    # plot the mandelbrot set
    fig, ax = plot_layout()
    plt.title("Mandelbrot set")
    plt.imshow(mandelbrot_set, extent=dims)
    plt.savefig("results/mandelbrot_{}_{}.png".format(grid_size, mandel_max_iter), dpi=1000)

    # save in true size
    # plt.imsave("results/mandelbrot_{}_{}_true_size.png", arr=mandelbrot_set, format='png')

    ###
    # comment out next part if you only want to calculate the integral and don't want to see the
    # convergence rate (takes relatively long to calculate)
    ###
    ########################################################################################

    # investigate the convergence
    mandelbrot_areas_conv = []
    for max_iter in range(mandel_max_iter):

        print("Evaluating max_iter {} of {} now...".format(max_iter, mandel_max_iter))

        # create mandelbrot set using max_iter
        mandelbrot_set_conv = mandelbrot(dims, grid_size, max_iter)

        # integrate mandelbrot set using max_iter
        surface = integrate_mandelbrot(mandelbrot_set_conv, dims, grid_size, max_iter, antithetic=True)
        mandelbrot_areas_conv.append(surface)

    # calculate A_js - A_is
    area_diffs = [(mandelbrot_area_conv - mandelbrot_area) for mandelbrot_area_conv in mandelbrot_areas_conv]
    max_iters = [max_iter for max_iter in range(mandel_max_iter)]

    # plot convergence rate of integral value
    fig2, ax2 = plot_layout()
    plt.title("Absolute difference in surface over number of iterations")
    plt.plot(max_iters, area_diffs)
    plt.xlabel("Iterations")
    plt.ylabel("Differences in surface")
    plt.savefig("results/mandelbrot_diffs_iter_{}_{}.png".format(grid_size, mandel_max_iter), dpi=1000)
    ########################################################################################

    plt.show()
    print("Done!")

    return


# execute file
if __name__ == "__main__":
    main()
