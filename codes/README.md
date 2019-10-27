## Todo

Run multiple sessions for the same environment to generate statistical results.
This is especially useful for the learning based system.
The initial condition, noises, and weight intializations may be different for each sessions.

We have to change the memory as a list. It is much faster than numpy append and remove.

-[ ] Plotting

## Environments

MRAC, Concurrent-learning MRAC, Concurrent Re-weighting MRAC


To run the entire scripts for re-generating all the results:
```
python main.py --all
```

First, we run a simulation for a standard MRAC:
```
python main.py --env mrac --log-dir log/base
```

To run simulation:
```
python main.py --env [mrac|clmrac|rlmrac] --algo sac --log-dir log/base
```

To plot the results (the argument is a directory):
```
python main.py --plot data/{yyyymmdd-hhmmss}
```

To get an animation of the results:
```
python main.py --anim-plot data/{...}
```

## Plot

```
python plot.py --all
```

```
python plot.py --compare data/{...} data/{...}
```

```
python plot.py --anim data/{...}
```



## Logger

The `Logger` module can be used as
