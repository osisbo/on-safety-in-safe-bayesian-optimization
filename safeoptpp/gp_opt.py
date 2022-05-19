"""
The MIT License (MIT)

Copyright (c) 2015 Felix Berkenkamp
Modifications copyright (c) 2022 Anonymous Author

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import print_function, absolute_import, division

from collections import Sequence
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import norm
from builtins import range

from .utilities import (plot_2d_gp, plot_3d_gp, plot_contour_gp,
                        linearly_spaced_combinations)
import logging
import math

__all__ = ['SafeOpt']


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters
    ----------
    gp: GPy Gaussian process
    fmin : float or list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted).
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.
    """

    def __init__(self, gp, fmin, beta=2, num_contexts=0, threshold=0,
                 scaling='auto'):
        """Initialization, see `GaussianProcessOptimization`."""
        super(GaussianProcessOptimization, self).__init__()

        if isinstance(gp, list):
            self.gps = gp
        else:
            self.gps = [gp]
        self.gp = self.gps[0]

        self.fmin = fmin
        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        if scaling == 'auto':
            dummy_point = np.zeros((1, self.gps[0].input_dim))
            self.scaling = [gpm.kern.Kdiag(dummy_point)[0] for gpm in self.gps]
            self.scaling = np.sqrt(np.asarray(self.scaling))
        else:
            self.scaling = np.asarray(scaling)
            if self.scaling.shape[0] != len(self.gps):
                raise ValueError("The number of scaling values should be "
                                 "equal to the number of GPs")

        self.threshold = threshold
        self._parameter_set = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts

        self._x = None
        self._y = None
        self._get_initial_xy()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        """Return the data within the GP models."""
        return self._x, self._y

    @property
    def t(self):
        """Return the time step (number of measurements)."""
        return self._x.shape[0]

    def _get_initial_xy(self):
        """Get the initial x/y data from the GPs."""
        self._x = self.gp.X
        y = [self.gp.Y]

        for gp in self.gps[1:]:
            if np.allclose(self._x, gp.X):
                y.append(gp.Y)
            else:
                raise NotImplemented('The GPs have different measurements.')

        self._y = np.concatenate(y, axis=1)

    def plot(self, n_samples, axis=None, figure=None, plot_3d=False,
             **kwargs):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        n_samples: int
            How many samples to use for plotting
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        figure: matplotlib figure
            Ignored if axis is already defined
        plot_3d: boolean
            If set to true shows a 3D plot for 2 dimensional data
        """
        # Fix contexts to their current values
        if self.num_contexts > 0 and 'fixed_inputs' not in kwargs:
            kwargs.update(fixed_inputs=self.context_fixed_inputs)

        true_input_dim = self.gp.kern.input_dim - self.num_contexts
        if true_input_dim == 1 or plot_3d:
            inputs = np.zeros((n_samples ** true_input_dim, self.gp.input_dim))
            inputs[:, :true_input_dim] = linearly_spaced_combinations(
                self.bounds[:true_input_dim],
                n_samples)

        if not isinstance(n_samples, Sequence):
            n_samples = [n_samples] * len(self.bounds)

        axes = []
        if self.gp.input_dim - self.num_contexts == 1:
            # 2D plots with uncertainty
            for gp, fmin in zip(self.gps, self.fmin):
                if fmin == -np.inf:
                    fmin = None
                ax = plot_2d_gp(gp, inputs, figure=figure, axis=axis,
                                fmin=fmin, **kwargs)
                axes.append(ax)
        else:
            if plot_3d:
                for gp in self.gps:
                    plot_3d_gp(gp, inputs, figure=figure, axis=axis, **kwargs)
            else:
                for gp in self.gps:
                    plot_contour_gp(gp,
                                    [np.linspace(self.bounds[0][0],
                                                 self.bounds[0][1],
                                                 n_samples[0]),
                                     np.linspace(self.bounds[1][0],
                                                 self.bounds[1][1],
                                                 n_samples[1])],
                                    figure=figure,
                                    axis=axis)

    def _add_context(self, x, context):
        """Add the context to a vector.

        Parameters
        ----------
        x : ndarray
        context : ndarray

        Returns
        -------
        x_extended : ndarray
        """
        context = np.atleast_2d(context)
        num_contexts = context.shape[1]

        x2 = np.empty((x.shape[0], x.shape[1] + num_contexts), dtype=float)
        x2[:, :x.shape[1]] = x
        x2[:, x.shape[1]:] = context
        return x2

    def _add_data_point(self, gp, x, y, context=None):
        """Add a data point to a particular GP.

        This should only be called on its own if you know what you're doing.
        This does not update the global data stores self.x and self.y.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points
        gp: instance of GPy.model.GPRegression
            If specified, determines the GP to which we add the data point
            to. Note that this should only be used if that data point is going
            to be removed again.
        """
        if context is not None:
            x = self._add_context(x, context)

        gp.set_XY(np.vstack([gp.X, x]),
                  np.vstack([gp.Y, y]))

    def add_new_data_point(self, x, y, context=None):
        """
        Add a new function observation to the GPs.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            x = self._add_context(x, context)

        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                # Add data to GP (context already included in x)
                self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)

    def _remove_last_data_point(self, gp):
        """Remove the last data point of a specific GP.

        This does not update global data stores, self.x and self.y.

        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP."""
        last_y = self._y[-1]

        for gp, yi in zip(self.gps, last_y):
            if not np.isnan(yi):
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

        self._x = self._x[:-1, :]
        self._y = self._y[:-1, :]


class SafeOpt(GaussianProcessOptimization):
    """A class for Safe Bayesian Optimization.

    This class implements the `SafeOpt` algorithm. It uses a Gaussian
    process model in order to determine parameter combinations that are safe
    with high probability. Based on these, it aims to both expand the set of
    safe parameters and to find the optimal parameters within the safe set.

    Parameters
    ----------
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    parameter_set: 2d-array
        List of parameters
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    lipschitz: list of floats
        The Lipschitz constant of the system, if None the GP confidence
        intervals are used directly.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.

    Examples
    --------
    >>> from safeopt import SafeOpt
    >>> from safeopt import linearly_spaced_combinations
    >>> import GPy
    >>> import numpy as np

    Define a Gaussian process prior over the performance

    >>> x = np.array([[0.]])
    >>> y = np.array([[1.]])
    >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)

    >>> bounds = [[-1., 1.]]
    >>> parameter_set = linearly_spaced_combinations([[-1., 1.]],
    ...                                              num_samples=100)

    Initialize the Bayesian optimization and get new parameters to evaluate

    >>> opt = SafeOpt(gp, parameter_set, fmin=[0.])
    >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    >>> performance = np.array([[1.]])
    >>> opt.add_new_data_point(next_parameters, performance)
    """

    def __init__(self, gp, parameter_set, fmin, beta_dict, lipschitz=None,
                 num_contexts=0, threshold=0, scaling='auto'):
        """Initialization, see `SafeOpt`."""

        # Overwrite beta init
        if beta_dict["style"] == "fiedler":
            print("Fiedler-bounds used!")
            B = beta_dict["B"]
            R = beta_dict["R"]
            delta = beta_dict["delta"]
            lamb = beta_dict["lambda"]  # gpr noise variance
            lamb_hat = max(1, lamb)
            beta = lambda t: B + \
                             (R / lamb ** 0.5) * math.sqrt(
                math.log(np.linalg.det(self.gp.kern.K(self.data[0], self.data[0])
                                       + lamb_hat * np.identity(len(self.data[0]))))
                - 2 * math.log(delta))

        elif beta_dict["style"] == "chowdhury":
            print("Chowdhury-bounds used!")
            B = beta_dict["B"]
            R = beta_dict["R"]
            delta = beta_dict["delta"]
            noise_variance = beta_dict["noise_variance"]
            card_D = beta_dict["card_D"]
            beta = lambda t: B + R * math.sqrt(
                2 * (card_D * math.log(1 + (noise_variance ** -1) * t * card_D *
                                       np.amax(np.diag(
                                           self.gp.kern.K(parameter_set, parameter_set)))) + 1 + math.log(
                    1 / delta)))

        elif beta_dict["style"] == "srinivas":
            print("Srinivas-bounds used!")
            B = beta_dict["B"]
            R = beta_dict["R"]
            delta = beta_dict["delta"]
            noise_variance = beta_dict["noise_variance"]
            card_D = beta_dict["card_D"]
            beta = lambda t: math.sqrt(2 * B + 300 * card_D * math.log(1 + (noise_variance ** -1) * t * card_D *
                                                                       np.amax(np.diag(self.gp.kern.K(parameter_set,
                                                                                                      parameter_set)))) * \
                                       math.log(t / delta) ** 3)

        else:
            print("No beta assigned! beta = 2 is used")
            beta = 2

        self.safety = beta_dict["safety"]
        self.index_x0 = beta_dict["index_x0"]

        super(SafeOpt, self).__init__(gp,
                                      fmin=fmin,
                                      beta=beta,
                                      num_contexts=num_contexts,
                                      threshold=threshold,
                                      scaling=scaling)

        if self.num_contexts > 0:
            context_shape = (parameter_set.shape[0], self.num_contexts)
            self.inputs = np.hstack((parameter_set,
                                     np.zeros(context_shape,
                                              dtype=parameter_set.dtype)))
            self.parameter_set = self.inputs[:, :-self.num_contexts]
        else:
            self.inputs = self.parameter_set = parameter_set

        self.liptschitz = lipschitz

        if self.liptschitz is not None:
            if not isinstance(self.liptschitz, list):
                self.liptschitz = [self.liptschitz] * len(self.gps)
            self.liptschitz = np.atleast_1d(
                np.asarray(self.liptschitz).squeeze())

        # Value intervals
        self.Q = np.empty((self.inputs.shape[0], 2 * len(self.gps)),
                          dtype=np.float)

        # Safe set
        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)
        self.S_x0 = self.S.copy()
        self.S_x0[self.index_x0] = 1

        # Switch to use confidence intervals for safety
        if lipschitz is None:
            self._use_lipschitz = False
        else:
            self._use_lipschitz = True

        # Set of expanders and maximizers
        self.G = self.S.copy()
        self.M = self.S.copy()

    @property
    def use_lipschitz(self):
        """
        Boolean that determines whether to use the Lipschitz constant.

        By default this is set to False, which means the adapted SafeOpt
        algorithm is used, that uses the GP confidence intervals directly.
        If set to True, the `self.lipschitz` parameter is used to compute
        the safe and expanders sets.
        """
        return self._use_lipschitz

    @use_lipschitz.setter
    def use_lipschitz(self, value):
        if value and self.liptschitz is None:
            raise ValueError('Lipschitz constant not defined')
        self._use_lipschitz = value

    @property
    def parameter_set(self):
        """Discrete parameter samples for Bayesian optimization."""
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set

        # Plotting bounds (min, max value
        self.bounds = list(zip(np.min(self._parameter_set, axis=0),
                               np.max(self._parameter_set, axis=0)))
        self.num_samples = [len(np.unique(self._parameter_set[:, i]))
                            for i in range(self._parameter_set.shape[1])]

    @property
    def context_fixed_inputs(self):
        """Return the fixed inputs for the current context."""
        n = self.gp.input_dim - 1
        nc = self.num_contexts
        if nc > 0:
            contexts = self.inputs[0, -self.num_contexts:]
            return list(zip(range(n, n - nc, -1), contexts))

    @property
    def context(self):
        """Return the current context variables."""
        if self.num_contexts:
            return self.inputs[0, -self.num_contexts:]

    @context.setter
    def context(self, context):
        """Set the current context and update confidence intervals.

        Parameters
        ----------
        context: ndarray
            New context that should be applied to the input parameters
        """
        if self.num_contexts:
            if context is None:
                raise ValueError('Need to provide value for context.')
            self.inputs[:, -self.num_contexts:] = context

    def update_confidence_intervals(self, context=None):
        """Recompute the confidence intervals form the GP.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        beta = self.beta(self.t)
        # print("Current beta for updating confidence intervals: ", beta)

        # Update context to current setting
        self.context = context

        # Iterate over all functions
        for i in range(len(self.gps)):
            # Evaluate acquisition function
            mean, var = self.gps[i].predict_noiseless(self.inputs)

            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            # Update confidence intervals
            self.Q[:, 2 * i] = mean - beta * std_dev
            self.Q[:, 2 * i + 1] = mean + beta * std_dev

    def compute_safe_set(self):
        """Compute only the safe set based on the current confidence bounds."""
        # Update safe set
        if self.safety == "modified":
            self.S[:] = np.all(self.Q[:, ::2] > self.fmin, axis=1)
        if self.safety == "lipschitz":
            # This implementation does only work for SafeOpt with just a performance constraint (and not additional constraints)!
            next_points_arr = np.logical_not(self.S)
            next_points_indices = np.nonzero(next_points_arr)[0]
            S_extended = np.logical_or(self.S, self.S_x0)
            safe_points_indices = np.nonzero(S_extended)[0]
            for index in safe_points_indices:
                delta_S = np.where(
                    self.Q[:, 0][index] - self.liptschitz[0] * np.linalg.norm(
                        self.inputs[index] - self.inputs[next_points_indices],
                        axis=1) >= self.fmin[0], 1,
                    0)
                self.S[next_points_indices] = np.logical_or(self.S[next_points_indices], delta_S)

    def compute_sets(self, full_sets=False):
        """
        Compute the safe set of points, based on current confidence bounds.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """
        beta = self.beta(self.t)
        # print("Current beta for computing safe set: ", beta)

        # Update safe set
        self.compute_safe_set()

        # Reference to confidence intervals
        l, u = self.Q[:, :2].T

        if not np.any(self.S):
            self.M[:] = False
            self.G[:] = False
            return

        # Set of possible maximisers
        # Maximizers: safe upper bound above best, safe lower bound
        self.M[:] = False
        self.M[self.S] = u[self.S] >= np.max(l[self.S])
        max_var = np.max(u[self.M] - l[self.M]) / self.scaling[0]

        # Optimistic set of possible expanders
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        self.G[:] = False

        # For the run of the algorithm we do not need to calculate the
        # full set of potential expanders:
        # We can skip the ones already in M and ones that have lower
        # variance than the maximum variance in M, max_var or the threshold.
        # Amongst the remaining ones we only need to find the
        # potential expander with maximum variance
        if full_sets:
            s = self.S
        else:
            # skip points in M, they will already be evaluated
            s = np.logical_and(self.S, ~self.M)

            # Remove points with a variance that is too small
            s[s] = (np.max((u[s, :] - l[s, :]) / self.scaling, axis=1) >
                    max_var)
            s[s] = np.any(u[s, :] - l[s, :] > self.threshold * beta, axis=1)

            if not np.any(s):
                # no need to evaluate any points as expanders in G, exit
                return

        def sort_generator(array):
            """Return the sorted array, largest element first."""
            return array.argsort()[::-1]

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)

        if not full_sets:
            # Sort, element with largest variance first
            sort_index = sort_generator(np.max(u[s, :] - l[s, :],
                                               axis=1))
        else:
            # Sort index is just an enumeration of all safe states
            sort_index = range(len(G_safe))

        for index in sort_index:
            if self.use_lipschitz:
                # Distance between current index point and all other unsafe
                # points
                d = cdist(self.inputs[s, :][[index], :],
                          self.inputs[~self.S, :])

                # Check if expander for all GPs
                for i in range(len(self.gps)):
                    # Skip evaluation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue
                    # Safety: u - L * d >= fmin
                    G_safe[index] = \
                        np.any(u[s, i][index] - self.liptschitz[i] * d >=
                               self.fmin[i])
                    # Stop evaluating if not expander according to one
                    # safety constraint
                    if not G_safe[index]:
                        break
            else:
                # Check if expander for all GPs
                for i, gp in enumerate(self.gps):
                    # Skip evlauation if 'no' safety constraint
                    if self.fmin[i] == -np.inf:
                        continue

                    # Add safe point with its max possible value to the gp
                    self._add_data_point(gp=gp,
                                         x=self.parameter_set[s, :][index, :],
                                         y=u[s, i][index],
                                         context=self.context)

                    # Prediction of previously unsafe points based on that
                    mean2, var2 = gp.predict_noiseless(self.inputs[~self.S])

                    # Remove the fake data point from the GP again
                    self._remove_last_data_point(gp=gp)

                    mean2 = mean2.squeeze()
                    var2 = var2.squeeze()
                    l2 = mean2 - beta * np.sqrt(var2)

                    # If any unsafe lower bound is suddenly above fmin then
                    # the point is an expander
                    G_safe[index] = np.any(l2 >= self.fmin[i])

                    # Break if one safety GP is not an expander
                    if not G_safe[index]:
                        break

            # Since we sorted by uncertainty and only the most
            # uncertain element gets picked by SafeOpt anyways, we can
            # stop after we found the first one
            if G_safe[index] and not full_sets:
                break

        # Update safe set (if full_sets is False this is at most one point
        self.G[s] = G_safe

    def get_new_query_point(self, ucb=False):
        """
        Compute a new point at which to evaluate the function.

        Parameters
        ----------
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')

        if ucb:
            max_id = np.argmax(self.Q[self.S, 1])
            x = self.inputs[self.S, :][max_id, :]
        else:
            # Get lower and upper bounds
            l = self.Q[:, ::2]
            u = self.Q[:, 1::2]

            MG = np.logical_or(self.M, self.G)
            value = np.max((u[MG] - l[MG]) / self.scaling, axis=1)
            x = self.inputs[MG, :][np.argmax(value), :]

        if self.num_contexts:
            return x[:-self.num_contexts]
        else:
            return x

    def optimize(self, context=None, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # Update confidence intervals based on current estimate
        self.update_confidence_intervals(context=context)

        # Update the sets
        if ucb:
            self.compute_safe_set()
        else:
            self.compute_sets()

        return self.get_new_query_point(ucb=ucb)

    def get_maximum(self, context=None):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context

        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        Notes
        -----
        Uses the current context and confidence intervals!
        Run update_confidence_intervals first if you recently added a new data
        point.
        """
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return nothing if there are no safe points
        if not np.any(self.S):
            return None

        l = self.Q[self.S, 0]

        max_id = np.argmax(l)
        return (self.inputs[self.S, :][max_id, :-self.num_contexts or None],
                l[max_id])

    def get_index_maximal_mean(self, context=None):
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return x0 if there are no safe points
        if not np.any(self.S):
            return self.index_x0

        mean, var = self.gps[0].predict_noiseless(self.inputs)
        S_indices = np.nonzero(self.S)[0]
        mean = mean[S_indices]
        max_mean_index_rel = np.argmax(mean)
        return S_indices[max_mean_index_rel]

    def get_index_maximal_u(self, context=None):
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return x0 if there are no safe points
        if not np.any(self.S):
            return self.index_x0

        u = self.Q[self.S, 1]
        S_indices = np.nonzero(self.S)[0]
        max_u_index_rel = np.argmax(u)

        return S_indices[max_u_index_rel]

    def get_index_maximal_l(self, context=None):
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return x0 if there are no safe points
        if not np.any(self.S):
            return self.index_x0

        l = self.Q[self.S, 0]
        S_indices = np.nonzero(self.S)[0]
        max_l_index_rel = np.argmax(l)

        return S_indices[max_l_index_rel]
