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
    from matplotlib.animation import FuncAnimation
    from collections import OrderedDict

    episodic = logging.load("data/rlcmrac/episodic.h5")

    fig, axes = plt.subplots(2, 1)
    time, cmd, state1, ref1, ref2, state2, control = [[] for _ in range(7)]
    lines = OrderedDict()
    lines['cmd'], = axes[0].plot([], [], 'k--')
    lines['state1'], = axes[0].plot([], [], 'r')
    lines['state2'], = axes[0].plot([], [], 'b')
    lines['ref1'], = axes[0].plot([], [], 'r--')
    lines['ref2'], = axes[0].plot([], [], 'b--')
    lines['memory'], = axes[1].plot([], [], 'g')
    lines['control'], = axes[1].plot([], [])

    max_action = episodic['action'].max()

    def init():
        [ax.set_xlim(0, episodic['time'].max()) for ax in axes]
        axes[0].set_ylim(-3, 3)
        axes[1].set_ylim(-300, 300)
        return lines.values()

    def update(frame):
        time.append(episodic['time'][frame])
        cmd.append(episodic['cmd'][frame])
        state1.append(episodic['state']['main_system'][frame][0])
        state2.append(episodic['state']['main_system'][frame][1])
        ref1.append(episodic['state']['reference_system'][frame][0])
        ref2.append(episodic['state']['reference_system'][frame][1])
        control.append(episodic['control']['MRAC'][frame])

        # print(episodic['memory']['t'][frame])
        time_action = episodic['memory']['t'][frame]
        action = 300 / max_action * episodic['action'][frame]

        episodic['memory']['t']

        lines['cmd'].set_data(time, cmd)
        lines['state1'].set_data(time, state1)
        lines['state2'].set_data(time, state2)
        lines['ref1'].set_data(time, ref1)
        lines['ref2'].set_data(time, ref2)
        lines['control'].set_data(time, control)

        lines['memory'].set_data(time_action, action)
        return lines.values()

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(len(episodic['time'])), interval=1
    )
    plt.show()
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
