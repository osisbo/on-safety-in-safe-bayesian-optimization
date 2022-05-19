import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
import numpy as np
import sys

dir_path = sys.argv[1]
kernel_lenghtscale = float(sys.argv[2])
B = float(sys.argv[3])

with open(dir_path + "/" + "result_dict.pickle", 'rb') as handle:
    result_dict = pickle.load(handle)

results_safeoptpp = result_dict["fiedler"]
results_safeopt = result_dict["chowdhury"]
results_losbo = result_dict["fiedler-lipschitz"]

current_regret_safeoptpp = results_safeoptpp[0]
current_regret_safeopt = results_safeopt[0]
current_regret_losbo = results_losbo[0]

aggregated_results_safeoptpp = np.zeros(shape=(100, 100))
aggregated_results_safeopt = np.zeros(shape=(100, 100))
aggregated_results_losbo = np.zeros(shape=(100, 100))

for i in range(10):
    for j in range(10):
        for k in range(100):
            aggregated_results_safeoptpp[i * 10 + j, k] += \
                current_regret_safeoptpp[i * 1000:(i + 1) * 1000][j * 100:(j + 1) * 100][k][3]
            aggregated_results_safeopt[i * 10 + j, k] += \
                current_regret_safeopt[i * 1000:(i + 1) * 1000][j * 100:(j + 1) * 100][k][3]
            aggregated_results_losbo[i * 10 + j, k] += \
                current_regret_losbo[i * 1000:(i + 1) * 1000][j * 100:(j + 1) * 100][k][3]

avg_safeoptpp = np.mean(aggregated_results_safeoptpp, axis=0)
upper_percentile_safeoptpp = np.percentile(aggregated_results_safeoptpp, 90, axis=0)
lower_percentile_safeoptpp = np.percentile(aggregated_results_safeoptpp, 10, axis=0)

avg_safeopt = np.mean(aggregated_results_safeopt, axis=0)
upper_percentile_safeopt = np.percentile(aggregated_results_safeopt, 90, axis=0)
lower_percentile_safeopt = np.percentile(aggregated_results_safeopt, 10, axis=0)

avg_losbo = np.mean(aggregated_results_losbo, axis=0)
upper_percentile_losbo = np.percentile(aggregated_results_losbo, 90, axis=0)
lower_percentile_losbo = np.percentile(aggregated_results_losbo, 10, axis=0)

x = np.linspace(0, 100, 100)

plt.plot(x, avg_safeoptpp, '.-', label="SafeOpt++")
plt.fill_between(x, lower_percentile_safeoptpp, upper_percentile_safeoptpp, alpha=0.2, label="SafeOpt++")

plt.plot(x, avg_safeopt, '.-', label="SafeOpt")
plt.fill_between(x, lower_percentile_safeopt, upper_percentile_safeopt, alpha=0.2, label="SafeOpt")

plt.plot(x, avg_losbo, '.-', label="LoSBO")
plt.fill_between(x, lower_percentile_losbo, upper_percentile_losbo, alpha=0.2, label="LoSBO")

plt.legend(loc="upper right", fontsize=13)
plt.xlabel(r"$t$")
plt.ylabel(r"Instantaneous regret $r_t$")
plt.grid()
plt.title(r"$l={}$, $\parallel f \parallel_k={}$".format(kernel_lenghtscale, B), fontsize=16)
plt.savefig("{}/losbo_{}_{}.pdf".format(dir_path, kernel_lenghtscale, B),
            dpi=1200)
plt.show()
