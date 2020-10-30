import numpy as np
import matplotlib.pyplot as plt
import time


def mandelbrot(dims, grid_size, mandel_max_iter):
    """Create mandelbrot set

    Args:
        grid_size -- length of grid in each direction
        mandel_max_iter -- maximum number of iterations per candidate number c

    Returns:
        mandelbrot_set -- matrix of number of iterations before escaping mandelbrot.
                          Equal to mandel_max_iter if in mandelbrot set
    """

    # create grid points to evaluate for mandelbrot set
    x_grid = np.linspace(dims[0], dims[1], grid_size)
    y_grid = np.linspace(dims[2], dims[3], grid_size)

    # create matrix to store number of iterations to escape the set
    # if it doesnt escape: iterations is set to 0
    mandelbrot_set = np.zeros((grid_size, grid_size))

    # loop over all coordinates
    for i, x in enumerate(x_grid):
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
    mandelbrot_set = np.loadtxt(fname)

    return mandelbrot_set

def save_mandelbot(fname, mandelbrot_set):
    """Save mandelbrot set """
    np.savetxt(fname, mandelbrot_set)

    return

def integrate_mandelbrot(mandelbrot_set, dims, grid_size, mandel_max_iter, mc_max_iter=10000):
    """Integrate the mandelbrot set using Monte Carlo method """

    hit = 0
    miss = 0

    print(np.amax(mandelbrot_set))

    for i in range(mc_max_iter):
        # throw a random number on the grid
        x_rand = np.random.randint(grid_size)
        y_rand = np.random.randint(grid_size)

        if mandelbrot_set[y_rand][x_rand] == mandel_max_iter - 1:
            hit += 1
        else:
            miss += 1

    print(hit, miss)
    # calculate total surface over which we integrate
    surface = (dims[1] - dims[0]) * (dims[3] - dims[2])

    # calculate are of mandelbrot
    area = hit / (miss + hit) * (surface)

    return area

def main():

    np.random.seed()

    dims = [-2, 0.6, -1, 1]
    grid_size = 1000
    mandel_max_iter = 100
    fname = "results/mandelbrot_{}_{}".format(grid_size, mandel_max_iter)

    # set to True if you want to calculate a new mandelbrot set
    # set to False if you want to load an existing mandelbrot set from fname
    calculate_new_mandelbrot = False

    if calculate_new_mandelbrot == True:
        time_start = time.time()

        # calculate the mandelbrot set and save to txt file
        mandelbrot_set = mandelbrot(dims, grid_size, mandel_max_iter)
        save_mandelbot(fname, mandelbrot_set)
        print("Time to calculate mandelbrot set: {} s".format(time.time() - time_start))
    else:

        # load mandelbrot set (only use this when not calculating a new mandelbrot set)
        mandelbrot_set = load_mandelbrot(fname)

    # calculate integral of mandelbrot set using Monte Carlo
    mandelbrot_area = integrate_mandelbrot(mandelbrot_set, dims, grid_size, mandel_max_iter)
    print("The integral of the mandelbrot set is {}".format(mandelbrot_area))

    fig, ax = plot_layout()
    plt.imshow(mandelbrot_set, extent=dims)
    plt.savefig("results/mandelbrot_{}_{}.png".format(grid_size, mandel_max_iter), dpi=1000)
    plt.show()
    print("Done!")

    return


# execute file
if __name__ == "__main__":
    main()
