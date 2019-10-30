import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import fym.logging as logging


def figure_1():
    """
    This figure compares the states, control inputs, the minimum eigenvalues of
    the information matrices of all methods.
    """
    BASE_DATA_DIR = "data"

    exp_list = os.listdir(BASE_DATA_DIR)
    exp_list.remove("tmp")

    def get_data(exp):
        path = os.path.join(BASE_DATA_DIR, exp, "episodic.h5")
        data = logging.load(path)
        return data

    formatting = {
        "command": dict(label="Command", color="k", linestyle="--"),
        "reference": dict(label="Reference", color="k", linestyle="-"),
        "mrac-nullagent": dict(
            label="MRAC", color="r", alpha=0.5, linestyle="-"),
        "fecmrac-nullagent": dict(label="FE-CMRAC", color="r", linestyle="-."),
        "rlcmrac-sac": dict(label="RL-CMRAC", color="b", linestyle="-")
    }

    # Draw common plots
    data = get_data(exp_list[0])
    plt.figure(num="state 1", figsize=[6.4, 4.8])
    plt.plot(
        data["time"], data["cmd"],
        **formatting["command"]
    )
    plt.plot(
        data["time"], data["state"]["reference_system"][:, 0],
        **formatting["reference"]
    )
    plt.ylabel(r"$x_1$")
    plt.xlabel("Time, sec")

    plt.figure(num="state 2", figsize=[6.4, 4.8])
    plt.plot(
        data["time"], data["state"]["reference_system"][:, 1],
        **formatting["reference"]
    )
    plt.ylabel(r"$x_2$")
    plt.xlabel("Time, sec")

    plt.figure(num="control")
    plt.ylabel(r"$u$")
    plt.xlabel("Time, sec")

    def general_figures(exp):
        data = get_data(exp)
        plt.figure(num="state 1")
        ax = plt.gca()
        ax.plot(
            data["time"], data["state"]["main_system"][:, 0],
            **formatting[exp],
        )

        plt.figure(num="state 2")
        plt.plot(
            data["time"], data["state"]["main_system"][:, 1],
            **formatting[exp],
        )

        plt.figure(num="control")
        plt.plot(
            data["time"], data["control"],
            **formatting[exp],
        )

    for exp in exp_list:
        general_figures(exp)

    exp_list.remove("mrac-nullagent")

    def eig_figures(exp, minmax):
        data = get_data(exp)
        if minmax == "min":
            eig = np.min(data["eigs"], axis=1)
        elif minmax == "max":
            eig = np.max(data["eigs"], axis=1)
        plt.plot(data["time"], eig, **formatting[exp])

    fig, (ax_min, ax_max) = plt.subplots(2, 1, num="Eigenvalues", sharex=True)

    plt.subplot(211)
    for exp in exp_list:
        eig_figures(exp, "min")
    plt.ylabel(r"Minimum $\lambda$")
    plt.legend()

    plt.subplot(212)
    for exp in exp_list:
        eig_figures(exp, "max")
    plt.ylabel(r"Maximum $\lambda$")
    plt.xlabel("Time, sec")

    # Saving
    def savefig(name):
        path = os.path.join(args.save_dir, name + "." + args.save_ext)
        plt.savefig(path, bbox_inches="tight")

    plt.figure(num="state 1")
    plt.legend()
    savefig("state-1")

    plt.figure(num="state 2")
    plt.legend()
    savefig("state-2")

    plt.figure(num="control")
    plt.legend()
    savefig("control")

    plt.figure(num="Eigenvalues")
    savefig("eigs")

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
    lines["cmd"], = axes[0].plot([], [], "k--")
    lines["state1"], = axes[0].plot([], [], "r")
    lines["state2"], = axes[0].plot([], [], "b")
    lines["ref1"], = axes[0].plot([], [], "r--")
    lines["ref2"], = axes[0].plot([], [], "b--")
    lines["memory"] = axes[1].stem(
        [0], [0],
        use_line_collection=True,
        linefmt="r",
    )
    lines["control"], = axes[1].plot([], [])

    max_action = episodic["action"].max()

    def init():
        [ax.set_xlim(0, episodic["time"].max()) for ax in axes]
        axes[0].set_ylim(-3, 3)
        axes[1].set_ylim(-300, 300)
        return lines.values()

    def to_segments(xs, ys):
        return [[[x, 0], [x, y]] for x, y in zip(xs, ys)]

    def update(frame):
        time.append(episodic["time"][frame])
        cmd.append(episodic["cmd"][frame])
        state1.append(episodic["state"]["main_system"][frame][0])
        state2.append(episodic["state"]["main_system"][frame][1])
        ref1.append(episodic["state"]["reference_system"][frame][0])
        ref2.append(episodic["state"]["reference_system"][frame][1])
        control.append(episodic["control"][frame])

        # print(episodic["memory"]["t"][frame])
        time_action = episodic["memory"]["t"][frame]
        action = 300 / max_action * episodic["action"][frame]
        segments = to_segments(time_action, action)

        episodic["memory"]["t"]

        lines["cmd"].set_data(time, cmd)
        lines["state1"].set_data(time, state1)
        lines["state2"].set_data(time, state2)
        lines["ref1"].set_data(time, ref1)
        lines["ref2"].set_data(time, ref2)
        lines["control"].set_data(time, control)

        lines["memory"].stemlines.set_segments(segments)
        return lines.values()

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, len(episodic["time"]), 10), interval=1
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
    lines["cmd"], = axes[0].plot([], [], "k--")
    lines["state1"], = axes[0].plot([], [], "r")
    lines["state2"], = axes[0].plot([], [], "b")
    lines["ref1"], = axes[0].plot([], [], "r--")
    lines["ref2"], = axes[0].plot([], [], "b--")
    lines["memory"] = axes[1].fill_between([], [])
    lines["control"], = axes[1].plot([], [])

    fill_option = dict(alpha=0.6, facecolor="r")

    control_max = episodic["control"].max()

    def init():
        [ax.set_xlim(0, episodic["time"].max()) for ax in axes]
        axes[0].set_ylim(-3, 3)
        axes[1].set_ylim(-300, 300)
        return lines.values()

    def update(frame):
        time.append(episodic["time"][frame])
        cmd.append(episodic["cmd"][frame])
        state1.append(episodic["state"]["main_system"][frame][0])
        state2.append(episodic["state"]["main_system"][frame][1])
        ref1.append(episodic["state"]["reference_system"][frame][0])
        ref2.append(episodic["state"]["reference_system"][frame][1])
        control.append(episodic["control"][frame])

        ta_idx = np.argmax(
            episodic["time"] > episodic["memory"]["time"][frame]
        )
        dist_time = episodic["time"][:ta_idx]
        dist_k = episodic["k"][:ta_idx]
        dist = [
            control_max * np.exp(-np.trapz(dist_k[i:], dist_time[i:]))
            for i in range(len(dist_time))
        ]

        episodic["memory"]["time"]

        lines["cmd"].set_data(time, cmd)
        lines["state1"].set_data(time, state1)
        lines["state2"].set_data(time, state2)
        lines["ref1"].set_data(time, ref1)
        lines["ref2"].set_data(time, ref2)
        lines["control"].set_data(time, control)

        lines["memory"].axes.collections.clear()
        lines["memory"].axes.fill_between(dist_time, dist, **fill_option)
        return lines.values()

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, len(episodic["time"]), 10), interval=1
    )
    plt.show()
    # anim.save("media/rlcmrac_action_anim.mp4")


def main(args):
    if args.all:
        figure_1()
        figure_2()
        figure_3()

    if args.num == 1:
        figure_1()
    elif args.num == 2:
        figure_2()
    elif args.num == 3:
        figure_3()
    else:
        raise ValueError(f"The figure {args.num} is not found")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--all", action="store_true")
    group.add_argument("-n", "--num", type=int)
    parser.add_argument("-s", "--save-dir", default="img")
    parser.add_argument("-e", "--save-ext", default="eps")
    args = parser.parse_args()
    main(args)
