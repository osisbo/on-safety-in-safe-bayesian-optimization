import GPy
import losbo
from operator import itemgetter
import gym
import numpy as np
import math
import statistics

def run_simulation(env_seed):

    env = gym.make("CartPole-v1")
    env.seed(env_seed)

    g = env.env.gravity
    mp = env.env.masspole
    mk = env.env.masscart
    lp = env.env.length

    # alter the parameters so that K is not "optimal" anymore
    mp = 2 * mp
    mk = 2 * mk
    lp = 2 * lp

    # Angle at which to fail the episode
    env.env.theta_threshold_radians = 12 * 2 * math.pi / 360
    env.env.x_threshold = 2.4

    a = g / (lp * (4.0 / 3 - mp / (mp + mk)))
    A = np.array([[0, 1, 0, 0],
                  [0, 0, a, 0],
                  [0, 0, 0, 1],
                  [0, 0, a, 0]])

    # input matrix
    b = -1 / (lp * (4.0 / 3 - mp / (mp + mk)))
    B = np.array([[0], [1 / mp + mk], [0], [b]])

    R = np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5 * np.eye(4, dtype=int)  # choose Q (weight for state)

    # get riccati solver
    from scipy import linalg

    # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
               np.dot(B.T, P))

    def apply_state_controller(K, x):
        # feedback controller
        u = -np.dot(K, x)  # u = -Kx
        if u > 0:
            return 1, u  # if force_dem > 0 -> move cart right
        else:
            return 0, u  # if force_dem <= 0 -> move cart left


    interval_K1 = [3 * K[0][0], 0.3 * K[0][0]]
    interval_K2 = [3 * K[0][1], 0.3 * K[0][1]]
    interval_K3 = [3 * K[0][2], 0.3 * K[0][2]]
    interval_K4 = [3 * K[0][3], 0.3 * K[0][3]]


    ####################################################################################################################

    def simulate_episode_testing(Ki):
        observation = env.reset()
        performance_list = []
        number_failures = 0
        # default episode length is 500
        for i in range(500):
            env.render()

            # get force direction (action) and force value (force)
            action, force = apply_state_controller(Ki, observation)

            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))

            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            observation, reward, done, info = env.step(action)
            performance_list.append(abs(observation[0]))

            if done:
                observation = env.reset()
                if i != 499:
                    number_failures += 1

        env.close()

        return np.expand_dims(np.array(max(performance_list)), axis=(0, 1))


    def simulate_episode_training(Ki):
        observation = env.reset()
        performance_list = []
        number_failures = 0
        # default episode length is 500
        for i in range(500):
            env.render()

            # get force direction (action) and force value (force)
            action, force = apply_state_controller(Ki, observation)

            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))

            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            observation, reward, done, info = env.step(action)
            performance_list.append(abs(observation[0]))

            if done:
                observation = env.reset()
                if i != 499:
                    number_failures += 1

        env.close()

        return np.expand_dims(np.array(max(performance_list)), axis=(0, 1))


    def transform_i_to_x(i):
        x1 = (i[0][0] - interval_K1[0]) / (interval_K1[1] - interval_K1[0])
        x2 = (i[0][1] - interval_K2[0]) / (interval_K2[1] - interval_K2[0])
        x3 = (i[0][2] - interval_K3[0]) / (interval_K3[1] - interval_K3[0])
        x4 = (i[0][3] - interval_K4[0]) / (interval_K4[1] - interval_K4[0])

        return np.expand_dims(np.array([x1, x2, x3, x4]), axis=0)


    def transform_x_to_i(x):
        i1 = x[0][0] * (interval_K1[1] - interval_K1[0]) + interval_K1[0]
        i2 = x[0][1] * (interval_K2[1] - interval_K2[0]) + interval_K2[0]
        i3 = x[0][2] * (interval_K3[1] - interval_K3[0]) + interval_K3[0]
        i4 = x[0][3] * (interval_K4[1] - interval_K4[0]) + interval_K4[0]

        return np.expand_dims(np.array([i1, i2, i3, i4]), axis=0)


    ########################################################################################################################

    L = 10
    number_iterations_optimization = 30
    grid_points_per_axis = 50
    noise_std = 0.1
    kernel_lengthscale = 0.5
    kernel_variance = 2.0

    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # Bounds on the inputs variables
    parameter_set = losbo.linearly_spaced_combinations(bounds, num_samples=grid_points_per_axis)

    i0_temp = K
    parameter_set_for_indexing = losbo.linearly_spaced_combinations(
        [(interval_K1[0], interval_K1[1]), (interval_K2[0], interval_K2[1]), (interval_K3[0], interval_K3[1]),
         (interval_K4[0], interval_K4[1])],
        num_samples=grid_points_per_axis)

    # find closest point to wanted safe seed in grid and use this closest point as starting point for optimization
    index_x0 = np.linalg.norm(parameter_set_for_indexing - i0_temp, axis=1).argmin()
    i0 = np.expand_dims(parameter_set_for_indexing[index_x0], axis=0)
    print("Initial controller parameters", i0)
    x0 = transform_i_to_x(i0)
    safety_threshold_exploration = -0.8
    safety_threshold_exploration_gp = 0
    y0 = -simulate_episode_training(i0) - safety_threshold_exploration
    print("Initial performance: ", y0)
    print("The safety threshold while optimizing is: ", safety_threshold_exploration_gp)
    print("Best performance possible is 0.8! Let's optimize!")

    card_D = grid_points_per_axis ** len(bounds)

    beta_dict = {"style": "fiedler", "B": 2, "R": noise_std, "delta": 0.01, "lambda": noise_std ** 2,
                 "noise_variance": noise_std ** 2, "card_D": card_D, "safety": "pure-lipschitz", "index_x0": index_x0,
                 "y0": y0}

    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=kernel_variance, lengthscale=kernel_lengthscale, ARD=False)
    gp = GPy.models.GPRegression(x0, y0, kernel, noise_var=noise_std ** 2)
    opt = losbo.SafeOpt(gp, parameter_set, lipschitz=L, fmin=[safety_threshold_exploration_gp], beta_dict=beta_dict)

    number_safety_violations = 0
    list_data_meas = []
    for i in range(number_iterations_optimization):
        print("Iteration: ", i)
        # Obtain next query point
        x_next, index_x = opt.optimize()
        i_next = transform_x_to_i(np.expand_dims(x_next, axis=0))
        print("Next controller parameters to try: ", i_next)
        # Get a measurement from the real system
        measured_cost = -simulate_episode_training(i_next) - safety_threshold_exploration
        y_meas = measured_cost
        list_data_meas.append((i_next, y_meas))
        print("Measured performance at iteration {}: ".format(i), y_meas)
        if y_meas < safety_threshold_exploration_gp:
            number_safety_violations += 1
            print("Measured safety violation!")
        # Add data to the GP model
        opt.add_new_data_point(np.expand_dims(x_next, axis=0), index_x, y_meas)

    # summarize results
    print("-" * 100)
    print("Optimization done! Number of safety violations: {}".format(number_safety_violations))
    data_opt = max(list_data_meas, key=itemgetter(1))
    y_opt = data_opt[1]
    i_opt = data_opt[0]

    performance_increase = y_opt - y0
    print("Increase in performance: {}".format(performance_increase))
    print("-" * 100)

    return performance_increase

if __name__ == "__main__":
    result_list = []
    for i in range(10):
        result_list.append(run_simulation(i))

    print("Average performance increase over simulations: ", statistics.mean([e[0][0] for e in result_list]))
