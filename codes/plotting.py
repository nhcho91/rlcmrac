import sys
import argparse
import matplotlib.pyplot as plt

import fym.logging as logging


def figure_1():
    data_mrac = logging.load("data/rlcmrac/history.h5")

    plt.figure(num='state 1', figsize=[6.4, 4.8])
    plt.plot(
        data_mrac['time'],
        data_mrac['cmd'],
        'k--',
    )
    plt.plot(
        data_mrac['time'],
        data_mrac['state']['reference_system'][:, 0],
        'k',
    )
    plt.plot(
        data_mrac['time'],
        data_mrac['state']['main_system'][:, 0],
        'r-.',
    )

    plt.figure(num='state 2', figsize=[6.4, 4.8])
    plt.plot(
        data_mrac['time'],
        data_mrac['state']['reference_system'][:, 1],
        'k',
    )
    plt.plot(
        data_mrac['time'],
        data_mrac['state']['main_system'][:, 1],
        'r-.',
    )

    plt.figure(num='control')
    plt.plot(data_mrac['time'], data_mrac['control']['MRAC'])

    plt.show()


def figure_2():
    """
    This figure creates an animation that shows the history of what action was
    executed during the simulation.
    """
    from matplotlib import animation

    data = logging.load("data/rlcmrac/history.h5")
    action_data = logging.load("data/rlcmrac/action_history.h5")

    fig, ax = plt.subplots()

    def init():
        ax.set_xlim(0, data['time'].max())
        ax.set_ylim(-2, 2)

    def update(frame):
        action = action_data[
        ax.axvspan
        return ln,

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=range(len(data['time'])),
        interval=50, blit=True
    )
    # anim.save("media/rlcmrac_action_anim.mp4")


def main(args):
    figure_2()
    return None

    if args.all:
        figure_1()

    if args.num == 1:
        figure_1()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--all", action="store_true")
    group.add_argument("-n", "--num", type=int)
    args = parser.parse_args()
    main(args)
