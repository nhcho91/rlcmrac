import sys
import matplotlib.pyplot as plt

from fym.utils import logger

for path in sys.argv[1:]:
    data = logger.load_dict_from_hdf5(sys.argv[1])

    plt.figure()
    plt.plot(data['time'], data['state']['main_system'][:, :2])

    plt.figure()
    plt.plot(data['time'], data['state']['main_system'][:, :2])


plt.show()
