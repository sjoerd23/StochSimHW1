import numpy as np
import matplotlib.pyplot as plt
import time

# NOT IN USE
def mandelbrot_map(z, c):
    """Definition of the mandelbrot_map: z_n+1 = (z_n)^2 + c """
    return z**2 + c

def mandelbrot(grid_size, iterations):
    """Create mandelbrot set

    Args:
        grid_size -- length of grid in each direction
        iterations -- number of iterations per mandelbrot number c

    Returns:
        x_mandelbrot - list of x coordinates of mandelbrot set
        y_mandelbrot - list of y coordinates of mandelbrot set
    """

    # save mandelbrot numbers to these lists
    x_mandelbrot = []
    y_mandelbrot = []

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
            for _ in range(iterations):

                # mandelbrot formula z_n+1 = (z_n)^2 + c
                z = z**2 + c
                if abs(z) >= 2:
                    break

            # if after iterations z < 2 --> member of mandelbrot set
            if abs(z) < 2:
                x_mandelbrot.append(x)
                y_mandelbrot.append(y)

    return np.array(x_mandelbrot), np.array(y_mandelbrot)


def plot_layout():
    """Standard plot layout for figures """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#e8e8e8', axisbelow=True)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    return fig, ax

def main():

    time_start = time.time()

    # retrieve the mandelbrot set
    x_mandelbrot, y_mandelbrot = mandelbrot(2000, 50)

    print(time.time() - time_start)

    print(len(x_mandelbrot))
    fig, ax = plot_layout()
    plt.scatter(x_mandelbrot, y_mandelbrot)
    plt.show()

    return


# execute file
if __name__ == "__main__":
    main()
