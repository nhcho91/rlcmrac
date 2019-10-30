import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging


def figure_1():
    # data = logging.load("data/fecmrac-nullagent/episodic.h5")
    data = logging.load("data/rlcmrac-sac/episodic.h5")

    plt.figure(num='state 1', figsize=[6.4, 4.8])
    plt.plot(
        data['time'],
        data['cmd'],
        'k--',
    )
    plt.plot(
        data['time'],
        data['state']['reference_system'][:, 0],
        'k',
    )
    plt.plot(
        data['time'],
        data['state']['main_system'][:, 0],
        'r--',
    )

    plt.figure(num='state 2', figsize=[6.4, 4.8])
    plt.plot(
        data['time'],
        data['state']['reference_system'][:, 1],
        'k',
    )
    plt.plot(
        data['time'],
        data['state']['main_system'][:, 1],
        'r--',
    )

    plt.figure(num='control')
    plt.plot(data['time'], data['control'])

    fig, axes = plt.subplots(2, 1, num='minimum eigenvalue', sharex=True)
    min_eig = np.min(data['eigs'], axis=1)
    # hidden_min_eig = np.min(data['hidden_eigs'], axis=1)
    max_eig = np.max(data['eigs'], axis=1)
    # hidden_max_eig = np.max(data['hidden_eigs'], axis=1)
    axes[0].plot(data['time'], min_eig)
    # axes[0].plot(data['time'], hidden_min_eig, '--')
    axes[1].plot(data['time'], max_eig)
    # axes[1].plot(data['time'], hidden_max_eig, '--')

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
    lines['memory'] = axes[1].stem(
        [0], [0],
        use_line_collection=True,
        linefmt='r',
    )
    lines['control'], = axes[1].plot([], [])

    max_action = episodic['action'].max()

    def init():
        [ax.set_xlim(0, episodic['time'].max()) for ax in axes]
        axes[0].set_ylim(-3, 3)
        axes[1].set_ylim(-300, 300)
        return lines.values()

    def to_segments(xs, ys):
        return [[[x, 0], [x, y]] for x, y in zip(xs, ys)]

    def update(frame):
        time.append(episodic['time'][frame])
        cmd.append(episodic['cmd'][frame])
        state1.append(episodic['state']['main_system'][frame][0])
        state2.append(episodic['state']['main_system'][frame][1])
        ref1.append(episodic['state']['reference_system'][frame][0])
        ref2.append(episodic['state']['reference_system'][frame][1])
        control.append(episodic['control'][frame])

        # print(episodic['memory']['t'][frame])
        time_action = episodic['memory']['t'][frame]
        action = 300 / max_action * episodic['action'][frame]
        segments = to_segments(time_action, action)

        episodic['memory']['t']

        lines['cmd'].set_data(time, cmd)
        lines['state1'].set_data(time, state1)
        lines['state2'].set_data(time, state2)
        lines['ref1'].set_data(time, ref1)
        lines['ref2'].set_data(time, ref2)
        lines['control'].set_data(time, control)

        lines['memory'].stemlines.set_segments(segments)
        return lines.values()

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, len(episodic['time']), 10), interval=1
    )
    plt.show()
    # anim.save("media/rlcmrac_action_anim.mp4")


def figure_3():
    """
    This figure creates an animation that shows the history of what action was
    executed during the simulation.
    """
    from matplotlib.animation import FuncAnimation
    from collections import OrderedDict

    episodic = logging.load("data/fecmrac-nullagent/episodic.h5")

    fig, axes = plt.subplots(2, 1)
    time, cmd, state1, ref1, ref2, state2, control = [[] for _ in range(7)]
    lines = OrderedDict()
    lines['cmd'], = axes[0].plot([], [], 'k--')
    lines['state1'], = axes[0].plot([], [], 'r')
    lines['state2'], = axes[0].plot([], [], 'b')
    lines['ref1'], = axes[0].plot([], [], 'r--')
    lines['ref2'], = axes[0].plot([], [], 'b--')
    lines['memory'] = axes[1].fill_between([], [])
    lines['control'], = axes[1].plot([], [])

    fill_option = dict(alpha=0.6, facecolor='r')

    control_max = episodic['control'].max()

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
        control.append(episodic['control'][frame])

        ta_idx = np.argmax(
            episodic['time'] > episodic['memory']['time'][frame]
        )
        dist_time = episodic['time'][:ta_idx]
        dist_k = episodic['k'][:ta_idx]
        dist = [
            control_max * np.exp(-np.trapz(dist_k[i:], dist_time[i:]))
            for i in range(len(dist_time))
        ]

        episodic['memory']['time']

        lines['cmd'].set_data(time, cmd)
        lines['state1'].set_data(time, state1)
        lines['state2'].set_data(time, state2)
        lines['ref1'].set_data(time, ref1)
        lines['ref2'].set_data(time, ref2)
        lines['control'].set_data(time, control)

        lines['memory'].axes.collections.clear()
        lines['memory'].axes.fill_between(dist_time, dist, **fill_option)
        return lines.values()

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, len(episodic['time']), 10), interval=1
    )
    plt.show()
    # anim.save("media/rlcmrac_action_anim.mp4")


def main(args):
    if args.all:
        figure_1()
        figure_2()

    if args.num == 1:
        figure_1()
    elif args.num == 2:
        figure_2()
    elif args.num == 3:
        figure_3()
    else:
        raise ValueError(f"The figure {args.num} is not found")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--all", action="store_true")
    group.add_argument("-n", "--num", type=int)
    args = parser.parse_args()
    main(args)
