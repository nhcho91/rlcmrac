"""
Run entire experiments and save thd data
```
python main.py
```
"""
import numpy as np
import sys
import ujson

import torch

from utils import load_spec

from fym import logging


def main(args):
    if args.env == 'mrac':
        import run_mrac
        run_mrac.main()
    elif args.env == 'rlcmrac':
        import run_rlcmrac
        run_rlcmrac.main()

    if args.plot is True:
        import plotting
        plotting.main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--all', action='store_true')
    group.add_argument('-e', '--env', choices=('mrac', 'rlcmrac'))
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='If the argument is empty, it will plot the most recent data in'
    )
    args = parser.parse_args()

    main(args)
