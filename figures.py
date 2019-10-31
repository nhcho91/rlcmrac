import sys
import os
import argparse
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

import fym.logging as logging


BASE_DATA_DIR = "data"
FORMATTING = {
    "command": dict(label="Command", color="k", linestyle="--"),
    "reference": dict(label="Reference", color="k", linestyle="-"),
    "mrac-nullagent": dict(
        label="MRAC", color="r", alpha=0.5, linestyle="-"),
    "fecmrac-nullagent": dict(label="FE-CMRAC", color="r", linestyle="-."),
    "rlcmrac-sac": dict(label="RL-CMRAC", color="b", linestyle="-")
}


def get_data(exp, base=BASE_DATA_DIR):
    path = os.path.join(base, exp, "episodic.h5")
    data = logging.load(path)
    return data


def figure_1():
    """
    This figure compares the states, control inputs, the minimum eigenvalues of
    the information matrices of all methods.
    """
    exp_list = os.listdir(BASE_DATA_DIR)
    exp_list.remove("tmp")

    # Draw common plots
    data = get_data(exp_list[0])
    plt.figure(num="state 1", figsize=[6.4, 4.8])
    plt.plot(
        data["time"], data["cmd"],
        **FORMATTING["command"]
    )
    plt.plot(
        data["time"], data["state"]["reference_system"][:, 0],
        **FORMATTING["reference"]
    )
    plt.ylabel(r"$x_1$")
    plt.xlabel("Time, sec")

    plt.figure(num="state 2", figsize=[6.4, 4.8])
    plt.plot(
        data["time"], data["state"]["reference_system"][:, 1],
        **FORMATTING["reference"]
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
            **FORMATTING[exp],
        )

        plt.figure(num="state 2")
        plt.plot(
            data["time"], data["state"]["main_system"][:, 1],
            **FORMATTING[exp],
        )

        plt.figure(num="control")
        plt.plot(
            data["time"], data["control"],
            **FORMATTING[exp],
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
        plt.plot(data["time"], eig, **FORMATTING[exp])

    fig, (ax_min, ax_max) = plt.subplots(2, 1, num="Eigenvalues", sharex=True)

    plt.subplot(211)
    plt.ticklabel_format(style="sci", scilimits=(0, 0), useOffset=True)
    for exp in exp_list:
        eig_figures(exp, "min")
    plt.ylabel(r"Minimum $\lambda$")
    plt.legend()

    plt.subplot(212)
    for exp in exp_list:
        eig_figures(exp, "max")
    plt.ylabel(r"Maximum $\lambda$")
    plt.xlabel("Time, sec")

    # Estimation error
    for exp in exp_list:
        data = get_data(exp)
        error = nla.norm(
            data["state"]["adaptive_system"] - data["real_param"],
            axis=1
        )
        plt.figure(num="Estimation error")
        plt.plot(data["time"], error, **FORMATTING[exp])
    plt.xlabel("Time, sec")
    plt.ylabel("Error norm")

    # Real parameter
    data = get_data(exp_list[0])
    plt.plot(data["time"], data["real_param"].squeeze())

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

    plt.figure(num="Estimation error")
    plt.legend()
    savefig("estim-error")

    plt.show()


def figure_2():
    """
    This figure creates an animation that shows the history of what action was
    executed during the simulation.
    """
    from matplotlib.animation import FuncAnimation
    from collections import OrderedDict

    exp_list = os.listdir(BASE_DATA_DIR)
    exp_list.remove("tmp")
    exp_list.remove("mrac-nullagent")

    data = {exp: get_data(exp) for exp in exp_list}

    fig, ax = plt.subplots(figsize=[16, 9])
    plt.xlabel("Time, sec")

    linekw = dict(linewidth=1, alpha=0.3)

    def make_line(label, *args):
        ln, = plt.plot([], [], label=label, *args, **linekw)
        return ln

    time = data["rlcmrac-sac"]["time"]
    lines = [
        (
            make_line("Command", "k--"),
            data["fecmrac-nullagent"]["cmd"]
        ),
        (
            make_line(r"$x_{r,1}$", "k"),
            data["fecmrac-nullagent"]["state"]["reference_system"][:, 0]
        ),
        (
            make_line(r"$x_{r,2}$", "k"),
            data["fecmrac-nullagent"]["state"]["reference_system"][:, 1]
        ),
        (
            make_line(r"FE-CMRAC $x_{1}$", "r-."),
            data["fecmrac-nullagent"]["state"]["main_system"][:, 0],
        ),
        (
            make_line(r"FE-CMRAC $x_{2}$", "r-."),
            data["fecmrac-nullagent"]["state"]["main_system"][:, 1],
        ),
        (
            make_line(r"RL-CMRAC $x_{1}$", "b-."),
            data["rlcmrac-sac"]["state"]["main_system"][:, 0],
        ),
        (
            make_line(r"RL-CMRAC $x_{2}$", "b-."),
            data["rlcmrac-sac"]["state"]["main_system"][:, 1],
        ),
    ]

    dist_lines = {
        "rlcmrac-sac": plt.vlines(
            [], [], [],
            colors=FORMATTING["rlcmrac-sac"]["color"],
            alpha=1
        ),
        "fecmrac-nullagent": plt.vlines(
            [], [], [],
            colors=FORMATTING["fecmrac-nullagent"]["color"],
            alpha=1
        ),
    }
    memory_time_diff = np.diff(data["fecmrac-nullagent"]["memory"]["time"])

    plt.legend()

    def to_segments(xs, ys):
        return [[[x, 0], [x, y]] for x, y in zip(xs, ys)]

    def init():
        ax.set_xlim(0, time.max())
        ax.set_ylim(-1, 1)
        return [line for line, _ in lines] + list(dist_lines.values())

    def update(frame):
        for line, line_data in lines:
            line.set_data(time[:frame], line_data[:frame])

        # Update FE-CMRAC
        if frame > 0 and memory_time_diff[frame-1] != 0:
            ta_idx = np.argmax(
                time > data["fecmrac-nullagent"]["memory"]["time"][frame]
            )
            dist_time = time[:ta_idx]
            dist_k = data["fecmrac-nullagent"]["k"][:ta_idx]
            dist = [
                -np.exp(-np.trapz(dist_k[i:], dist_time[i:]))
                for i in range(len(dist_time))
            ]
            segments = to_segments(dist_time, dist)
            dist_lines["fecmrac-nullagent"].set_segments(segments)

        # Update RL-CMRAC
        dist_time = data["rlcmrac-sac"]["memory"]["t"][frame]
        dist = 100 * data["rlcmrac-sac"]["dist"][frame]
        segments = to_segments(dist_time, dist)
        dist_lines["rlcmrac-sac"].set_segments(segments)
        return [line for line, _ in lines] + list(dist_lines.values())

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=range(0, len(time), 10), interval=10
    )
    # plt.show()
    anim.save(os.path.join(args.save_dir, "dist-movie.mp4"))


def main(args):
    if args.all:
        figure_1()
        figure_2()

    if args.num == 1:
        figure_1()
    elif args.num == 2:
        figure_2()
    else:
        raise ValueError(f"The figure {args.num} is not found")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--all", action="store_true")
    group.add_argument("-n", "--num", type=int)
    parser.add_argument("-s", "--save-dir", default="img")
    parser.add_argument("-e", "--save-ext", default="png")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)
