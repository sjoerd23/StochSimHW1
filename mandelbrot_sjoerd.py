import numpy as np
import matplotlib.pyplot as plt
import time


def mandelbrot(grid_size, n_iterations):
    """Create mandelbrot set

    Args:
        grid_size -- length of grid in each direction
        n_iterations -- number of n_iterations per mandelbrot number c

    Returns:
        x_mandelbrot - list of x coordinates of mandelbrot set
        y_mandelbrot - list of y coordinates of mandelbrot set
    """

    # save all calculated mandelbrot numbers to these lists
    x_mandelbrot = []
    y_mandelbrot = []

    # store the amount of iterations it takes for a number to escape the set
    # if it doesnt escape: iterations is set to 0
    iterations_mandelbrot = []

    # check these points for mandelbrot numbers
    x_grid = np.linspace(-2, 0.6, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)

    # loop over all coordinates
    for x in x_grid:
        for y in y_grid:

            # create imaginary number c = a + bi
            c = complex(x, y)

            # max size of mandelbrot set is 2
            z = 0
            i = 0
            for i in range(n_iterations):

                # mandelbrot formula z_n+1 = (z_n)^2 + c
                z = z**2 + c
                if abs(z) >= 2:
                    break

            # save the number of iterations. 0 denotes mandelbrot set number
            # if after n_iterations z < 2 --> member of mandelbrot set
            if abs(z) < 2:
                iterations_mandelbrot.append(0)
            else:
                iterations_mandelbrot.append(i + 1)

            x_mandelbrot.append(x)
            y_mandelbrot.append(y)

    return np.array(x_mandelbrot), np.array(y_mandelbrot), np.array(iterations_mandelbrot)


def plot_layout():
    """Standard plot layout for figures """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#e8e8e8', axisbelow=True)
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(False)

    return fig, ax

def save_mandelbot(fname, x_mandelbrot, y_mandelbrot, iterations_mandelbrot):
    """Save mandelbrot set """
    np.savetxt(fname + "_x", x_mandelbrot)
    np.savetxt(fname + "_y", y_mandelbrot)
    np.savetxt(fname + "_iter", iterations_mandelbrot)

    return

def load_mandelbrot(fname):
    """Loads mandelbrot set """
    x_mandelbrot = np.loadtxt(fname + "_x")
    y_mandelbrot = np.loadtxt(fname + "_y")
    iterations_mandelbrot = np.loadtxt(fname + "_iter")

    return x_mandelbrot, y_mandelbrot, iterations_mandelbrot

def main():

    grid_size = 1000
    n_iterations = 50
    fname = "results/mandelbrot_{}_{}".format(grid_size, n_iterations)

    # set to True if you want to calculate a new mandelbrot set
    # set to False if you want to load an existing mandelbrot set from "results/"
    calculate_new_mandelbrot = False

    if calculate_new_mandelbrot == True:
        time_start = time.time()

        # calculate the mandelbrot set
        x_mandelbrot, y_mandelbrot, iterations_mandelbrot = mandelbrot(grid_size, n_iterations)

        # save mandelbrot set (only use this when calculating a new mandelbrot set)
        save_mandelbot(fname, x_mandelbrot, y_mandelbrot, iterations_mandelbrot)
        print("Time to calculate mandelbrot set: {} s".format(time.time() - time_start))

    else:
        time_start = time.time()

        # load mandelbrot set (only use this when not calculating a new mandelbrot set)
        x_mandelbrot, y_mandelbrot, iterations_mandelbrot = load_mandelbrot(fname)
        print("Time to load mandelbrot set: {} s".format(time.time() - time_start))

    # plot mandelbrot set (can take a while, because of the coloring of each coordinate)
    fig, ax = plot_layout()
    ax.set_xlim(-2, 0.5)
    ax.set_ylim(-1, 1)
    plt.scatter(x_mandelbrot, y_mandelbrot, c=iterations_mandelbrot)

    # saving take a lot of time, so only do this when you really want to keep the image.
    # Otherwise plt.show() is faster
    # print("Saving figure...")
    # plt.savefig("results/mandelbrot_{}_{}.png".format(grid_size, n_iterations), dpi=500)
    plt.show()
    print("Done!")

    return


# execute file
if __name__ == "__main__":
    main()
