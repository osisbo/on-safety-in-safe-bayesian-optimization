import datetime
import pickle
import numpy as np
import scipy.stats
import os
import pathlib

store_path = str(pathlib.Path(__file__).parent.resolve())
safety_threshold_exploration = 0.0
safety_threshold_x0 = 0.5
delta = 0.01
num_functions = 100
num_random_seeds = 100
number_iterations_optimization = 100
grid_points_per_axis = 50

safety_style = "lipschitz"

# Measurement noise
noise_var = 0.05 ** 2

kernel_variance = 2.0

# Parameters for estimating the Lipschitz constant
sample_interval = [0, 1]
number_sample_points = 50
number_iterations_lipschitz = 1000

# Bounds on the inputs variable
bounds = [(0., 1.), (0., 1.)]
# Produce the discretized grid for the inputs

#######################################################################################################################
root_path = store_path
experiment_data = ["fiedler", "chowdhury"]

dicts_paths_list = [name for name in
                    os.listdir(root_path) if
                    os.path.isdir(root_path + "/" + name)]

result_dict = {}
for style in experiment_data:
    result_dict[style] = [[], [], [], 0, [], [], [], [], [], [], [], 0, [], 0]
for name in dicts_paths_list:
    with open(root_path + "/" + name + "/" + "result_dict.pickle", 'rb') as handle:
        result_dict_i = pickle.load(handle)
        for style in experiment_data:
            result_dict[style][0] = result_dict[style][0] + result_dict_i[style][0]
            result_dict[style][1] = result_dict[style][1] + result_dict_i[style][1]
            result_dict[style][2] = result_dict[style][2] + result_dict_i[style][2]
            result_dict[style][3] += result_dict_i[style][3]
            result_dict[style][4] = result_dict[style][4] + result_dict_i[style][4]
            result_dict[style][5] = result_dict[style][5] + result_dict_i[style][5]
            result_dict[style][6] = result_dict[style][6] + result_dict_i[style][6]
            result_dict[style][7] = result_dict[style][7] + result_dict_i[style][7]
            result_dict[style][8] = result_dict[style][8] + result_dict_i[style][8]
            result_dict[style][9] = result_dict[style][9] + result_dict_i[style][9]
            result_dict[style][10] = result_dict[style][10] + result_dict_i[style][10]
            result_dict[style][11] += result_dict_i[style][11]
            result_dict[style][12] = result_dict[style][12] + result_dict_i[style][12]
            result_dict[style][13] += result_dict_i[style][13]

log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_path = store_path + "/" + log_time
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

print("-" * 100)
print("Cleaned overall comparison:")
print("-" * 100)

with open(dir_path + "/" + "safeopt_results.txt", "w") as f:
    print("-" * 100, file=f)
    print("Cleaned overall comparison:", file=f)
    print("-" * 100, file=f)

for counter, style in enumerate(experiment_data):
    average_regret_list = result_dict[style][0]
    average_regret_sum = 0
    for r in average_regret_list:
        r = r[3]
        if isinstance(r, np.ndarray):
            r = r[0]
        average_regret_sum += r

    final_regret_list = result_dict[style][1]
    final_regret_sum = 0
    for f in final_regret_list:
        if isinstance(f, np.ndarray):
            f = f[0]
        final_regret_sum += f

    f0_star_list = result_dict[style][2]
    f0_star_sum = 0
    for fs in f0_star_list:
        if isinstance(fs, np.ndarray):
            fs = fs[0]
        f0_star_sum += fs

    average_mean_regret_list = result_dict[style][4]
    average_mean_regret_sum = 0
    for r in average_mean_regret_list:
        r = r[3]
        if isinstance(r, np.ndarray):
            r = r[0]
        average_mean_regret_sum += r

    final_mean_regret_list = result_dict[style][5]
    final_mean_regret_sum = 0
    for f in final_mean_regret_list:
        if isinstance(f, np.ndarray):
            f = f[0]
        final_mean_regret_sum += f

    average_upper_regret_list = result_dict[style][6]
    average_upper_regret_sum = 0
    for r in average_upper_regret_list:
        r = r[3]
        if isinstance(r, np.ndarray):
            r = r[0]
        average_upper_regret_sum += r

    final_upper_regret_list = result_dict[style][7]
    final_upper_regret_sum = 0
    for f in final_upper_regret_list:
        if isinstance(f, np.ndarray):
            f = f[0]
        final_upper_regret_sum += f

    average_lower_regret_list = result_dict[style][8]
    average_lower_regret_sum = 0
    for r in average_lower_regret_list:
        r = r[3]
        if isinstance(r, np.ndarray):
            r = r[0]
        average_lower_regret_sum += r

    final_lower_regret_list = result_dict[style][9]
    final_lower_regret_sum = 0
    for f in final_lower_regret_list:
        if isinstance(f, np.ndarray):
            f = f[0]
        final_lower_regret_sum += f

    print("Average regret overall ({0}): {1}".format(style, average_regret_sum / len(average_regret_list)))
    print("Average final regret ({0}): {1}".format(style, final_regret_sum / len(final_regret_list)))
    print("Encountered exceptions due to no available safe points ({0}): {1}".format(style, result_dict[style][
        3] / number_iterations_optimization))
    print("Average value of f0_star ({0}): {1}".format(style, f0_star_sum / len(f0_star_list)))
    print("Average mean regret overall ({0}): {1}".format(style,
                                                          average_mean_regret_sum / len(average_mean_regret_list)))
    print("Average final mean regret ({0}): {1}".format(style, final_mean_regret_sum / len(final_mean_regret_list)))
    print("Average upper regret overall ({0}): {1}".format(style, average_upper_regret_sum / len(
        average_upper_regret_list)))
    print("Average final upper regret ({0}): {1}".format(style,
                                                         final_upper_regret_sum / len(final_upper_regret_list)))
    print("Average lower regret overall ({0}): {1}".format(style, average_lower_regret_sum / len(
        average_lower_regret_list)))
    print("Average final lower regret ({0}): {1}".format(style,
                                                         final_lower_regret_sum / len(final_lower_regret_list)))
    print("Number of real overall safety violations ({0}): {1}".format(style, result_dict[style][11]))
    print("Number of measured overall safety violations ({0}): {1}".format(style, result_dict[style][13]))

    with open(dir_path + "/" + "safeopt_results.txt", "a") as f:
        print("Average regret overall ({0}): {1}".format(style, average_regret_sum / len(average_regret_list)),
              file=f)
        print("Average final regret ({0}): {1}".format(style, final_regret_sum / len(final_regret_list)), file=f)
        print("Encountered exceptions due to no available safe points ({0}): {1}".format(style, result_dict[style][
            3] / number_iterations_optimization),
              file=f)
        print("Average value of f0_star ({0}): {1}".format(style, f0_star_sum / len(f0_star_list)), file=f)

        print("Average mean regret overall ({0}): {1}".format(style, average_mean_regret_sum / len(
            average_mean_regret_list)), file=f)
        print("Average final mean regret ({0}): {1}".format(style,
                                                            final_mean_regret_sum / len(final_mean_regret_list)),
              file=f)
        print("Average upper regret overall ({0}): {1}".format(style, average_upper_regret_sum / len(
            average_upper_regret_list)), file=f)
        print("Average final upper regret ({0}): {1}".format(style,
                                                             final_upper_regret_sum / len(final_upper_regret_list)),
              file=f)
        print("Average lower regret overall ({0}): {1}".format(style, average_lower_regret_sum / len(
            average_lower_regret_list)), file=f)
        print("Average final lower regret ({0}): {1}".format(style,
                                                             final_lower_regret_sum / len(final_lower_regret_list)),
              file=f)
        print("Number of real overall safety violations ({0}): {1}".format(style, result_dict[style][11]), file=f)
        print("Number of measured overall safety violations ({0}): {1}".format(style, result_dict[style][13]), file=f)

p_value = scipy.stats.ttest_ind(result_dict["fiedler"][1], result_dict["chowdhury"][1]).pvalue[0][0]
print("p-value cleaned comparison (based on sample regret): {0}".format(p_value))

with open(dir_path + "/" + "safeopt_results.txt", "a") as f:
    print("p-value cleaned comparison (based on sample regret): {0}".format(p_value), file=f)
    print("-" * 100, file=f)
    print("Used Hyperparameters:", file=f)
    print("-" * 100, file=f)
    print("number_iterations_lipschitz: ", number_iterations_lipschitz, file=f)
    print("Safety style: ", safety_style, file=f)
    print("safety_threshold_exploration: ", safety_threshold_exploration, file=f)
    print("safety_threshold_x0: ", safety_threshold_x0, file=f)
    print("delta: ", delta, file=f)
    print("num_functions: ", num_functions, file=f)
    print("num_random_seeds: ", num_random_seeds, file=f)
    print("number_iterations_optimization: ", number_iterations_optimization, file=f)
    print("grid_points_per_axis: ", grid_points_per_axis, file=f)
    print("noise_var: ", noise_var, file=f)
    print("kernel_variance: ", kernel_variance, file=f)
    print("sample_interval: ", sample_interval, file=f)
    print("number_sample_points: ", number_sample_points, file=f)
    print("bounds: ", bounds, file=f)

# store result data
with open(dir_path + "/" + 'result_dict.pickle', 'wb') as handle:
    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
