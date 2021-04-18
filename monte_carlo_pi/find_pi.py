import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

def is_in_circle(x, y, r, center):
    """
    check if sampled point is inside the circle
    """
    # the point is inside the circle if
    # (x-x_center)^2 + (y-y_center)^2 < r^2
    r_pt = (x - center[0])**2 + (y - center[1])**2
    r_pt = np.sqrt(r_pt)
    # note that the case of equality doesn't
    # make a difference, since r_pt is a function
    # of continuous random variables X and Y
    if r_pt <= r:
        return True
    else:
        return False

def sample_points(s, n_points=None):
    """
    Sample x ~ X and y ~ Y from the uniform 
    distribution [0, s]
    """
    xs, ys = uniform(low=0.0, high=s, size=(2, n_points))
    
    return xs, ys

def approximate_pi(n_in_circle, n_total):
    """
    Return approximate value for pi based
    on the number of points sampled so far
    """
    # compute the ratio of points inside the circle
    # to total number of points
    ratio = n_in_circle / n_total

    # this ratio is equivalent to pi/4
    pi = 4 * ratio

    return pi

def simulate_pi(n_points=1000, s=1.0):
    """
    Given a number of points to sample,
    estimate the value of pi
    """
    # the radius of the inscribed circle 
    # is half the length of the square
    r = s / 2

    # we can then sample x, y pairs as specific values
    # of the uniform random variables X, Y on [0, s]
    # the bottom left corner of the square is assumed
    # to be the origin
    xs, ys = sample_points(s, n_points)

    # count how many sampled points landed inside
    # the circle
    n_in_circle = 0
    for x, y in zip(xs, ys):
        if is_in_circle(x, y, r, center=(s/2,s/2)):
            n_in_circle += 1

    pi = approximate_pi(n_in_circle, n_points)

    return pi

def animate_pi(s=1.0, thre=1e-4, n_max=1e6):
    """
    Watch the approximation converge to its familiar
    value in real time, given an accuracy threshold
    for the value of pi. 
    """
    # create plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    # generate data
    xs, ys = sample_points(s, int(n_max))

    # initialize plots
    scatter = ax.scatter([], [], 
                         color='cornflowerblue', 
                         marker='o',
                         s=20
                         )
    text = ax.text(x=0.8, y=0.1, 
                   s='', 
                   fontsize=12,
                   ha='center', 
                   va='center',
                   bbox=dict(
                       boxstyle='square', 
                       facecolor='white',
                       edgecolor='silver',
                       alpha=0.8
                       )
                   )

    # wrap init, frame data and update functions
    # in partial functions so they 
    init = partial(init_fig, fig=fig, ax=ax, scatter=scatter, text=text)
    update = partial(update_frame, scatter=scatter, text=text)
    fd = partial(frame_data, xs=xs, ys=ys)

    # make animation
    ani = FuncAnimation(
            fig=fig,
            func=update,
            frames=fd,
            init_func=init,
            interval=1,
            repeat_delay=2000,
            blit=True
            )

    plt.show()

def init_fig(fig, ax, scatter, text):
    """
    draw the initial background for the figure
    """
    # set axis and plot titles
    ax.set_title('$\pi$ sampling', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    # plot the circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = (s/2)*np.cos(theta) + (s/2)
    y_circle = (s/2)*np.sin(theta) + (s/2)
    ax.plot(x_circle, y_circle,
            color='firebrick',
            linewidth=3.0)

    # plot the square
    ax.plot([0,s], [0,0], color='black', linewidth=3.0)
    ax.plot([0,s], [s,s], color='black', linewidth=3.0)
    ax.plot([0,0], [0,s], color='black', linewidth=3.0)
    ax.plot([s,s], [0,s], color='black', linewidth=3.0)

    # give axis a bit of extra space
    ds = 0.005
    ax.axis([0-ds, s+ds, 0-ds, s+ds])

    # return the plots
    return scatter, text


def frame_data(xs, ys):
    """
    return data for each frame in the animation
    """
    # start a count of number of points in the circle
    n_in_circle = 0
    plot_xs = []
    plot_ys = []
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        plot_xs.append(x)
        plot_ys.append(y)
        if is_in_circle(x, y, r=s/2, center=(s/2,s/2)):
            n_in_circle += 1
        # total sampled points = i+1
        pi = approximate_pi(n_in_circle, i+1)
        pi_string = f'$\hat\pi$={pi:.6f}'
        # the yield function allows for another
        # function to iterate over update_fig
        yield (plot_xs, plot_ys, pi_string)

def update_frame(frame, scatter, text):
    """
    update the scatter plot and the current pi value
    """
    xs, ys, pi_string = frame
    points = [ [x, y] for x, y in zip(xs, ys) ]
    scatter.set_offsets(points)
    text.set_text(pi_string)

    return scatter, text

if __name__=='__main__':
    # let the side length of the square be length 1
    s = 1.0

    # animate the simulation to watch 
    # our approximate pi converge to pi!
    animate_pi(s=s, thre=1e-4)
    
    # # let's sample 100000 points
    # n_points = 100000
    # 
    # # run the simulation to approximate pi
    # pi = simulate_pi(n_points, s)
    # print('pi is approximately ', pi)
    
