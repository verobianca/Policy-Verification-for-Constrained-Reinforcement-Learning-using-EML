# Author: Mattia Silvestri

"""
    Utility script with methods and classes for the RL environments.
"""

import gym
import numpy as np
from ortools.sat.python import cp_model

########################################################################################################################


class PLSInstance:
    """
    Create an instance of the Partial Latin Square Constrained Problem with n numbers, one-hot encoded as
    an NxNxN numpy array. Each cell of the PLS represents a variable whose i-th bit is raised if number i is assigned.
    """
    def __init__(self, n=10, leave_columns_domains=False):
        # Problem dimension
        self.n = n
        # Problem instance
        self.square = np.zeros((n, n, n), dtype=np.int8)
        # Variables domains
        self._init_var_domains()
        self.remove_columns_domains = not leave_columns_domains

    def copy(self):
        """
        Create an instance which is equal to the current one.
        :return: PLSInstance.
        """

        obj = PLSInstance()
        obj.n = self.n
        obj.square = self.square.copy()
        obj.remove_columns_domains = self.remove_columns_domains
        return obj

    def _init_var_domains(self):
        """
        A method to initialize variables domains to [0, N]
        :return:
        """
        # (n, n) variables; n possible values; the index represents the value; 1 means removed from the domain
        self.domains = np.zeros(shape=(self.n, self.n, self.n), dtype=np.int8)

    def set_square(self, square, forward=False):
        """
        Public method to set the square.
        :param square: the square as numpy array of shape (dim, dim, dim)
        :param forward: True if you want to apply forward checking
        :return: True if the assignement is feasible, False otherwise
        """

        self.square = square
        feas = self._check_constraints()
        if feas and forward:
            self._init_var_domains()
            self._forward_checking()

        return feas

    def _check_constraints(self):
        """
        Check that all PLS constraints are consistent.
        :return: True if constraints are consistent, False otherwise.
        """
        multiple_var = np.sum(self.square, axis=2)
        rows_fail = np.sum(self.square, axis=1)
        cols_fail = np.sum(self.square, axis=0)

        if np.sum(multiple_var > 1) > 0:
            return False

        if np.sum(rows_fail > 1) > 0:
            return False

        if np.sum(cols_fail > 1) > 0:
            return False

        return True

    def check_constraints_type(self):
        """
        Check that all PLS constraints are consistent.
        :return: a list of three integers, one for each constraint type (multiple assignment, row violation,
        columns violation), representing violiations metric.
        """
        # How many times a value has been assigned to the same variable
        multiple_var = np.sum(self.square, axis=2)
        # How many times a value appears in the same row
        rows_fail = np.sum(self.square, axis=1)
        # How many times a value appears in the same columns
        cols_fail = np.sum(self.square, axis=0)

        constraint_type = [0, 0, 0]

        # How many times there is a multiple or lacking assignment
        constraint_type[0] = np.sum(multiple_var != 1)

        # How many equals values are there in a single row?
        constraint_type[1] = np.sum(rows_fail != 1)

        # How many equals values are there in a single column?
        constraint_type[2] = np.sum(cols_fail != 1)

        return constraint_type

    def get_assigned_variables(self):
        """
        Return indexes of assigned variables.
        :return: a numpy array containing indexes of assigned variables.
        """
        return np.argwhere(np.sum(self.square, axis=2) == 1)

    def _forward_checking(self):
        """
        Method to update variables domain with forward_checking
        :return:
        """

        for i in range(self.n):
            for j in range(self.n):
                # Find assigned value to current variable
                assigned_val = np.argwhere(self.square[i, j] == 1)
                assigned_val = assigned_val.reshape(-1)
                # Check if a variable is assigned
                if len(assigned_val) != 0:
                    # Current variable is already assigned -> domain is empty
                    self.domains[i, j] = np.ones(shape=(self.n,))
                    # Remove assigned value to same row and column variables domains
                    for id_cols in range(self.n):
                        self.domains[i, id_cols, assigned_val] = 1
                    if self.remove_columns_domains:
                        for id_row in range(self.n):
                            self.domains[id_row, j, assigned_val] = 1

    def assign(self, cell_x, cell_y, num):
        """
        Variable assignment.
        :param cell_x: x coordinate of a square cell
        :param cell_y: y coordinate of a square cell
        :param num: value assigned to cell
        :return: True if the assignment is consistent, False otherwise
        """

        # Create a temporary variable so that you can undo inconsistent assignment
        tmp_square = self.square.copy()

        if num > self.n-1 or num < 0:
            raise ValueError("Allowed values are in [0,{}]".format(self.n))
        else:
            self.square[cell_x, cell_y, num] += 1

        if not self._check_constraints():
            self.square = tmp_square.copy()
            return False

        return True

    def unassign(self, cell_x, cell_y):
        """
        Variable unassignment
        :param cell_x: x coordinate of a square cell
        :param cell_y: y coordinare of a square cell
        :return:
        """

        var = self.square[cell_x, cell_y].copy()
        assigned_val = np.argmax(var)
        self.square[cell_x, cell_y, assigned_val] = 0

    def visualize(self):
        """
        Visualize PLS.
        :return:
        """
        vals_square = np.argmax(self.square, axis=2) + np.sum(self.square, axis=2)

        print(vals_square)

########################################################################################################################


class PLSSolver:
    def __init__(self, board_size, square):
        """
        Class to build a PLS solver.
        :param board_size: number of variables
        :param square: numpy array with decimal assigned values
        """

        self.board_size = board_size

        # Create solver
        self.model = cp_model.CpModel()

        # Creates the variables.
        assigned = []
        for i in range(0, board_size ** 2):
            if square[i] > 0:
                assigned.append(self.model.NewIntVar(square[i], square[i], 'x%i' % i))
            else:
                assigned.append(self.model.NewIntVar(1, board_size, 'x%i' % i))

        # Creates the constraints.
        # All numbers in the same row must be different.
        for i in range(0, board_size ** 2, board_size):
            self.model.AddAllDifferent(assigned[i:i+board_size])

        # all numbers in the same column must be different
        for j in range(0, board_size):
            colmuns = []
            for idx in range(j, board_size ** 2, board_size):
                colmuns.append(assigned[idx])

            self.model.AddAllDifferent(colmuns)

        self.vars = assigned.copy()

    def solve(self):
        """
        Find a feasible solution.
        :return: True if a feasible solution was found, 0 otherwise
        """
        # create the solver
        solver = cp_model.CpSolver()
        # set time limit to 30 seconds
        solver.parameters.max_time_in_seconds = 30.0

        # solve the model
        status = solver.Solve(self.model)

        return status == cp_model.FEASIBLE


########################################################################################################################

class PLSEnv(gym.Env):
    """
    Gym wrapper for the PLS.
    Attributes:
        dim: int; PLS size.
        instance: PLSInstance; the PLS instance.
    """

    # This is a gym.Env variable that is required by garage for rendering
    metadata = {
        "render.modes": ["ascii"]
    }

    def __init__(self, dim):
        super(PLSEnv, self).__init__()

        self._dim = dim
        self.action_space = gym.spaces.Discrete(dim ** 3)
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.int8, shape=(dim ** 3,))
        self._instance = PLSInstance(n=dim)
        self.sol = 0
        self.first_solved = -1

    @property
    def dim(self):
        """
        The PLS size.
        :return: int; the PLS size.
        """
        return self._dim

    def step(self, action, episode=0, sparse_reward=True):

        """
        A step in the environment.
        :param action: int; integer corresponding to the PLS cell and value to assign.
        :return: numpy.array, float, boolean, dict; observations, reward, end of episode flag and additional info.
        """
        assert 0 <= action < self.action_space.n, "Out of actions space"

        x_coor, y_coor, val = np.unravel_index(action, shape=(self._dim, self._dim, self._dim))

        feasible = self._instance.assign(cell_x=x_coor, cell_y=y_coor, num=val)
        count_assigned_vars = np.sum(self._instance.square)
        solved = False
        reward = 0.0

        if feasible:
            if sparse_reward == False:
                reward = 1.0
            if count_assigned_vars == self._dim ** 2:
                done = True
                solved = True
                if sparse_reward:
                    reward = 5.0
                else:
                    reward = float(self._dim ** 2)
                self.sol += 1
                if self.sol == 1 and episode!=0:
                    print('First solved at episode: {}'.format(episode))
                    self.first_solved = episode

            else:
                done = False
        else:
            if sparse_reward:
                reward = -1.0
            else:
                reward = -float(self._dim ** 2)
            done = True

        obs = self._instance.square.reshape(-1)

        # Create dictionary with some useful information
        info = dict()
        info['Feasible'] = feasible
        info['Num. assigned vars'] = count_assigned_vars
        info['Solved'] = solved

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment.
        :return: numpy.array; the observations.
        """
        self._instance = PLSInstance(n=self._dim)
        # obs = self._instance.square.reshape(-1, 1)
        obs = self._instance.square.reshape(-1)

        return obs

    def render(self, mode="human"):
        """
        Visualize the PLS assignments.
        :param mode:
        :return:
        """
        self._instance.visualize()


########################################################################################################################
