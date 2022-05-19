from __future__ import print_function, division, absolute_import
import GPy
import numpy as np
import safeoptpp as safeopt
import scipy
import math
import time
import os
import datetime
import pickle
import multiprocessing as mp
import sys
import pathlib

store_path = str(pathlib.Path(__file__).parent.resolve())


def calculate_L_nom(lengthscale, noise_std, B_nom):
    grid_points_per_axis = 50

    # Nominal RKHS norm for the functions to use
    B_nom = B_nom

    # Measurement noise
    noise_var = noise_std ** 2

    kernel_variance = 2.0

    kernel_lengthscale = lengthscale

    # Parameters for estimating the Lipschitz constant
    sample_interval = [0, 1]
    number_sample_points = 50
    number_iterations_lipschitz = 1000

    # Bounds on the inputs variable
    bounds = [(0., 1.), (0., 1.)]
    # Produce the discretized grid for the inputs
    parameter_set = safeopt.linearly_spaced_combinations(bounds, num_samples=grid_points_per_axis)

    def sample_fun():
        return safeopt.utilities.sample_function_with_fixed_B(kernel, bounds, noise_var, grid_points_per_axis,
                                                              B_nom=B_nom)

    def get_lipschitz_constant(fun_gen, parameter_set, number_iterations, bounds, number_sample_points):
        gradient_list = []
        for _ in range(number_iterations):
            fun = fun_gen()
            fun_true = lambda x: fun(x, noise=False).flatten()
            grad_x, grad_y = np.gradient(fun_true(parameter_set).reshape(50, 50),
                                         (bounds[0][1] - bounds[0][0]) / number_sample_points)
            grad_vectors = np.concatenate((grad_x.reshape(50, 50, 1), grad_y.reshape(50, 50, 1)), axis=2)
            gradient_list.append(np.max(np.linalg.norm(grad_vectors, axis=2)))
        m = max(gradient_list)
        return m

    # Define Kernel:
    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=kernel_variance, lengthscale=kernel_lengthscale, ARD=False)

    # Compute Lipschitz constant if needed (can also be manually set if it was already computed before for used function class)
    L_temp = get_lipschitz_constant(fun_gen=sample_fun, parameter_set=parameter_set,
                                    number_iterations=number_iterations_lipschitz, bounds=bounds,
                                    number_sample_points=number_sample_points)

    return math.ceil(L_temp)


#######################################################################################################################


def run_exp(num_functions, factor_L, factor_B, lengthscale, noise_std, B, L_nom, used_random_seed):
    safety_threshold_exploration = 0.0
    safety_threshold_x0 = 0.5
    delta = 0.01
    num_random_seeds = 10
    number_iterations_optimization = 100
    grid_points_per_axis = 50

    # Nominal RKHS norm for the functions to use
    B_nom = B

    safety_style = "lipschitz"

    # Measurement noise
    noise_var = noise_std ** 2

    kernel_variance = 2.0

    kernel_lengthscale = lengthscale

    # Parameters for estimating the Lipschitz constant
    sample_interval = [0, 1]
    number_sample_points = 50
    number_iterations_lipschitz = 1000

    # Bounds on the inputs variable
    bounds = [(0., 1.), (0., 1.)]
    # Produce the discretized grid for the inputs
    parameter_set = safeopt.linearly_spaced_combinations(bounds, num_samples=grid_points_per_axis)
    card_D = grid_points_per_axis ** len(bounds)

    def get_fun_with_safe_grid_seeds_np():
        job_done = False
        while not job_done:
            fun = safeopt.utilities.sample_function_with_fixed_B(kernel, bounds, noise_var, grid_points_per_axis,
                                                                 B_nom=B_nom)
            function_values = fun(parameter_set, noise=False).flatten()
            filter_arr = np.where(function_values > safety_threshold_x0, True, False)
            if np.sum(filter_arr) > num_random_seeds:
                job_done = True
                indices_filter_arr = np.nonzero(filter_arr)[0]
                indices_chosen = np.random.randint(len(indices_filter_arr), size=num_random_seeds)
                indices_safe_seeds = indices_filter_arr[indices_chosen]
        return fun, indices_safe_seeds, function_values

    def get_reachable_maximum_np(grid, function_values, index_x0, L, threshold):
        safe_set_arr = np.zeros(grid.shape[0])
        safe_set_arr[index_x0] = 1
        safe_set_extended = True
        while safe_set_extended:
            safe_points_indices = np.nonzero(safe_set_arr)[0]
            # safe_set_arr[:] = 0
            next_points_arr = np.logical_not(safe_set_arr)
            next_points_indices = np.nonzero(next_points_arr)[0]
            safe_set_extended = False
            for index_nsp in safe_points_indices:
                delta_S = np.where(
                    function_values[index_nsp] - L * np.linalg.norm(grid[index_nsp] - grid[next_points_indices],
                                                                    axis=1) >= threshold, 1,
                    0)
                safe_set_arr[next_points_indices] = np.logical_or(safe_set_arr[next_points_indices], delta_S)
                if np.sum(delta_S) > 0:
                    safe_set_extended = True
        reachable_maximum = function_values[np.where(safe_set_arr == 1)].max()
        x0_star = grid[function_values[np.where(safe_set_arr == 1)].argmax()]
        return x0_star, reachable_maximum

    def timer(start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    training_start_time = time.time()

    # Define Kernel:
    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=kernel_variance, lengthscale=kernel_lengthscale, ARD=False)

    experiment_data = ["fiedler", "chowdhury"]

    present_process = mp.current_process()
    worker_id = present_process._identity[0]
    np.random.seed(used_random_seed)

    L_ass = factor_L * L_nom
    print("L_ass: ", L_ass)

    B_ass = factor_B * B_nom
    print("B_ass: ", B_ass)

    result_dict = {}
    for style in experiment_data:
        result_dict[style] = [[], [], [], 0, [], [], [], [], [], [], [], 0, [], 0]
    for i in range(num_functions):
        # Get function sample und corresponding set of safe seeds
        fun, safe_indices, function_values = get_fun_with_safe_grid_seeds_np()

        for j, index_x0 in enumerate(safe_indices):
            print("Function number: ", i, " | ", "Seed number: ", j)
            x0_star, f0_star = get_reachable_maximum_np(grid=parameter_set, function_values=function_values,
                                                        index_x0=index_x0, L=L_nom,
                                                        threshold=safety_threshold_exploration)
            x0 = parameter_set[index_x0].reshape(1, len(bounds))

            for style in experiment_data:
                y0 = fun(x0)
                gp = GPy.models.GPRegression(x0, y0, kernel, noise_var=noise_var)
                beta_dict = {"style": style, "B": B_ass, "R": noise_var ** 0.5, "delta": 0.01, "lambda": 1,
                             "noise_variance": noise_var, "card_D": card_D, "safety": safety_style,
                             "index_x0": index_x0}
                opt = safeopt.SafeOpt(gp, parameter_set, fmin=[safety_threshold_exploration], beta_dict=beta_dict,
                                      lipschitz=L_ass)

                for k in range(number_iterations_optimization):
                    # Obtain next query point
                    try:
                        x_next = opt.optimize()
                        # Get a measurement from the real system
                        y_meas = fun(x_next)
                        # Add this to the GP model
                        opt.add_new_data_point(x_next, y_meas)
                        y_real = fun(x_next, noise=False)
                        if y_real < safety_threshold_exploration:
                            result_dict[style][11] += 1
                        # Check for measured safety violations
                        if y_meas < safety_threshold_exploration:
                            result_dict[style][13] += 1
                        # Here, we now look at the true function value of the best observed point (changed from before)
                        index_maximum = np.argmax(opt.data[1])
                        x_max = opt.data[0][index_maximum]
                        current_maximum = fun(x_max, noise=False)
                        # print("Current maximum: ", current_maximum)
                        current_regret = f0_star - current_maximum
                        # print("Current regret: ", current_regret)

                        index_maximal_mean = opt.get_index_maximal_mean()
                        x_maximal_mean = parameter_set[index_maximal_mean].reshape(1, len(bounds))
                        current_maximal_mean = fun(x_maximal_mean, noise=False)
                        current_regret_with_mean = f0_star - current_maximal_mean
                        # print("current_regret_with_mean: ", current_regret_with_mean)

                        index_maximal_u = opt.get_index_maximal_u()
                        x_maximal_u = parameter_set[index_maximal_u].reshape(1, len(bounds))
                        current_maximal_u = fun(x_maximal_u, noise=False)
                        current_regret_with_u = f0_star - current_maximal_u
                        # print("current_regret_with_u: ", current_regret_with_u)

                        index_maximal_l = opt.get_index_maximal_l()
                        x_maximal_l = parameter_set[index_maximal_l].reshape(1, len(bounds))
                        current_maximal_l = fun(x_maximal_l, noise=False)
                        current_regret_with_l = f0_star - current_maximal_l
                        # print("current_regret_with_l: ", current_regret_with_l)

                        result_dict[style][0].append([i, j, k, current_regret])
                        result_dict[style][4].append([i, j, k, current_regret_with_mean])
                        result_dict[style][6].append([i, j, k, current_regret_with_u])
                        result_dict[style][8].append([i, j, k, current_regret_with_l])
                        result_dict[style][10].append([i, j, k, (x_next, y_real)])
                        result_dict[style][12].append([i, j, k, (x_next, y_meas)])

                        if k == number_iterations_optimization - 1:
                            result_dict[style][1].append(current_regret)
                            result_dict[style][2].append(f0_star)
                            result_dict[style][5].append(current_regret_with_mean)
                            result_dict[style][7].append(current_regret_with_u)
                            result_dict[style][9].append(current_regret_with_l)

                    except EnvironmentError:
                        result_dict[style][3] += 1
                        # print("Current maximum: ", current_maximum)
                        # try:
                        if k == 0:
                            current_maximum = fun(x0, noise=False)
                            current_regret = f0_star - current_maximum

                            current_maximal_mean = fun(x0, noise=False)
                            current_regret_with_mean = f0_star - current_maximal_mean

                            current_maximal_u = fun(x0, noise=False)
                            current_regret_with_u = f0_star - current_maximal_u

                            current_maximal_l = fun(x0, noise=False)
                            current_regret_with_l = f0_star - current_maximal_l

                            x_next = x0
                            y_real = fun(x_next, noise=False)
                            y_meas = y0

                        # print("Current regret: ", current_regret)
                        result_dict[style][0].append([i, j, k, current_regret])
                        result_dict[style][4].append([i, j, k, current_regret_with_mean])
                        result_dict[style][6].append([i, j, k, current_regret_with_u])
                        result_dict[style][8].append([i, j, k, current_regret_with_l])
                        result_dict[style][10].append([i, j, k, (x_next, y_real)])
                        result_dict[style][12].append([i, j, k, (x_next, y_meas)])

                        if k == number_iterations_optimization - 1:
                            result_dict[style][1].append(current_regret)
                            result_dict[style][2].append(f0_star)
                            result_dict[style][5].append(current_regret_with_mean)
                            result_dict[style][7].append(current_regret_with_u)
                            result_dict[style][9].append(current_regret_with_l)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print()
    dir_path = store_path + "/" + log_time + "-" + str(factor_L) + "-" + str(factor_B) + "-" + str(
        lengthscale) + "-" + str(noise_std) + "-" + str(B) + "-" + str(worker_id)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print("-" * 100)
    print("Cleaned comparison:")
    print("-" * 100)

    with open(dir_path + "/" + "safeopt_results.txt", "w") as f:
        print("-" * 100, file=f)
        print("Cleaned comparison:", file=f)
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
            print("Number of measured overall safety violations ({0}): {1}".format(style, result_dict[style][13]),
                  file=f)

    # Calculate p-value for the means of the average final regret samples from using the Fiedler- and Chowdhury-Bounds
    experiment_time = timer(training_start_time, time.time())

    p_value = scipy.stats.ttest_ind(result_dict["fiedler"][1], result_dict["chowdhury"][1]).pvalue[0][0]
    print("p-value cleaned comparison (based on sample regret): {0}".format(p_value))
    print('Experiment time: {}'.format(experiment_time + '\n'))

    with open(dir_path + "/" + "safeopt_results.txt", "a") as f:
        print("p-value cleaned comparison (based on sample regret): {0}".format(p_value), file=f)
        print('Experiment time: {}'.format(experiment_time + '\n'), file=f)
        print("-" * 100, file=f)
        print("Used Hyperparameters:", file=f)
        print("-" * 100, file=f)
        print("L_nom: ", L_nom, file=f)
        print("L_ass: ", L_ass, file=f)
        print("number_iterations_lipschitz: ", number_iterations_lipschitz, file=f)
        print("Safety style: ", safety_style, file=f)
        print("safety_threshold_exploration: ", safety_threshold_exploration, file=f)
        print("safety_threshold_x0: ", safety_threshold_x0, file=f)
        print("delta: ", delta, file=f)
        print("num_functions: ", num_functions, file=f)
        print("num_random_seeds: ", num_random_seeds, file=f)
        print("number_iterations_optimization: ", number_iterations_optimization, file=f)
        print("grid_points_per_axis: ", grid_points_per_axis, file=f)
        print("B_nom: ", B_nom, file=f)
        print("B_ass: ", B_ass, file=f)
        print("noise_var: ", noise_var, file=f)
        print("kernel_variance: ", kernel_variance, file=f)
        print("kernel_lengthscale: ", kernel_lengthscale, file=f)
        print("sample_interval: ", sample_interval, file=f)
        print("number_sample_points: ", number_sample_points, file=f)
        print("bounds: ", bounds, file=f)
        print("used_random_seed: ", used_random_seed, file=f)

    # store result data
    with open(dir_path + "/" + 'result_dict.pickle', 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    num_functions = 10
    num_workers = 1
    factor_L = 1
    factor_B = 100
    lengthscale = float(sys.argv[1])
    B_nom = float(sys.argv[2])
    use_predefined_safe_seeds = int(sys.argv[3])
    noise_std = 0.05
    predefined_safe_seeds_02_2 = [1652857066]
    predefined_safe_seeds_02_20 = [1652857101]
    predefined_safe_seeds_05_2 = [1652857116]
    predefined_safe_seeds_05_20 = [1652857137]
    L_02_2 = 13
    L_02_20 = 121
    L_05_2 = 6
    L_05_20 = 57


    def build_seed_list():
        seed_list = []
        for j in range(num_workers):
            added = False
            while not added:
                used_random_seed = int(time.time()) + j
                if used_random_seed not in seed_list:
                    seed_list.append(used_random_seed)
                    added = True
        return seed_list


    pool = mp.Pool(num_workers)

    if use_predefined_safe_seeds:
        if lengthscale == 0.2 and B_nom == 2:
            seed_list = predefined_safe_seeds_02_2
        elif lengthscale == 0.2 and B_nom == 20:
            seed_list = predefined_safe_seeds_02_20
        elif lengthscale == 0.5 and B_nom == 2:
            seed_list = predefined_safe_seeds_05_2
        elif lengthscale == 0.5 and B_nom == 20:
            seed_list = predefined_safe_seeds_05_20
        else:
            seed_list = build_seed_list()
    else:
        seed_list = build_seed_list()

    if lengthscale == 0.2 and B_nom == 2:
        L_nom = L_02_2
    elif lengthscale == 0.2 and B_nom == 20:
        L_nom = L_02_20
    elif lengthscale == 0.5 and B_nom == 2:
        L_nom = L_05_2
    elif lengthscale == 0.5 and B_nom == 20:
        L_nom = L_05_20
    else:
        L_nom = calculate_L_nom(lengthscale, noise_std, B_nom)

    for i in range(num_workers):
        pool.apply_async(run_exp,
                         args=(
                             num_functions, factor_L, factor_B, lengthscale, noise_std, B_nom,
                             L_nom, seed_list[i],))

    pool.close()
    pool.join()
