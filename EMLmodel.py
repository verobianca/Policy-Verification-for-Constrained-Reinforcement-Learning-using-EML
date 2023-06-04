import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from eml.backend import cplex_backend
from eml.net.embed import encode
from eml.net.reader import keras_reader


class EMLmodel:

    def __init__(self, savepath, dim):
        self.savepath = savepath
        self.dim = dim

    def convert_keras_net(self, knet):
        net = keras_reader.read_keras_sequential(knet)
        return net

    def delete_softmax(self, model):
        # create new model and replace softmax with linear activation
        model2 = Sequential()
        model2.add(Dense(32, activation='relu', input_shape=(self.dim ** 3,), name='common_body_layer_0'))
        for layer in model.layers:
            if layer.name.startswith("common_body_layer") and layer.name != "common_body_layer_0":
                model2.add(Dense(32, activation='relu', name=layer.name))
            if layer.name == "policy":
                model2.add(Dense(self.dim ** 3, activation="linear", name='policy'))

        # copy weights
        model2.set_weights(model.get_weights())
        return model2

    def load_keras_net(self):
        model = tf.keras.models.load_model(self.savepath)
        model2 = self.delete_softmax(model)
        return model2

    def set_variables(self):
        # variables
        self.X_vars = []  # input to the net (state in array form)
        self.solution_vars = []  # feasible solution
        self.Y_vars = []  # output of the net
        self.nextX_vars = []  # next_state in array form

        for i in range(self.dim**3):
            self.X_vars.append(self.model.integer_var(0, 1, name='input_{}'.format(i)))
            self.solution_vars.append(self.model.integer_var(0, 1, name='solution_{}'.format(i)))
            self.Y_vars.append(self.model.continuous_var(lb=-float('inf'), name='output_{}'.format(i)))
            self.nextX_vars.append(self.model.integer_var(0, 1, name='next_state_{}'.format(i)))

        x1 = np.array(self.X_vars)
        self.state = x1.reshape((self.dim, self.dim, self.dim))  # state in matrix form

        s = np.array(self.solution_vars)
        self.solution = s.reshape((self.dim, self.dim, self.dim))  # solution in matrix form

        x2 = np.array(self.nextX_vars)
        self.next_state = x2.reshape((self.dim, self.dim, self.dim))  # next_state in matrix form

        self.action = self.model.integer_var(0, self.dim**3-1, name='action')  # action variable

    def set_constraints(self):
        # One-hot encoding constraint
        for row in range(self.dim):
            for col in range(self.dim):
                # for the current state
                pos_state = self.state[row, col]
                self.model.add_constraint(self.model.sum(pos_state) <= 1)

                # for the solution
                pos_sol = self.solution[row, col]
                self.model.add_constraint(self.model.sum(pos_sol) <= 1)

                # for the next state
                pos_next = self.next_state[row, col]
                self.model.add_constraint(self.model.sum(pos_next) <= 1)

        # Feasibility constraint
        tras_state = np.transpose(self.state, (1, 0, 2))
        tras_solution = np.transpose(self.solution, (1, 0, 2))
        for i in range(self.dim):
            row = self.state[i]
            col = tras_state[i]
            row_sol = self.solution[i]
            col_sol = tras_solution[i]

            for j in range(self.dim):
                combinations = []  # for each combination of two position in the row/col
                for z1 in range(self.dim):
                    for z2 in range(self.dim):
                        if z1 != z2:
                            if (z1, z2) not in combinations and (z2, z1) not in combinations:
                                # constraints for the state
                                # it constrains all value in a row to be different
                                self.model.add(self.model.logical_and((self.model.sum(row[z1]) == 1), (self.model.sum(row[z2]) == 1),
                                                                      (row[z1, j] == 1)) <= (row[z2, j] != 1))
                                # it constrains all value in a column to be different
                                self.model.add(self.model.logical_and((self.model.sum(col[z1]) == 1), (self.model.sum(col[z2]) == 1),
                                                                      (col[z1, j] == 1)) <= (col[z2, j] != 1))

                                # constraints for the solution
                                # it constrains all value in a row to be different
                                self.model.add(self.model.logical_and((self.model.sum(row_sol[z1]) == 1), (self.model.sum(row_sol[z2]) == 1),
                                                                      (row_sol[z1, j] == 1)) <= (row_sol[z2, j] != 1))
                                # it constrains all value in a column to be different
                                self.model.add(self.model.logical_and((self.model.sum(col_sol[z1]) == 1), (self.model.sum(col_sol[z2]) == 1),
                                                                      (col_sol[z1, j] == 1)) <= (col_sol[z2, j] != 1))

                                combinations.append((z1, z2))

        # the current state must not be a solution
        self.model.add(self.model.sum(self.nextX_vars) <= self.dim ** 2 - 1)
        # the solution must have 9 assigned number
        self.model.add(self.model.sum(self.solution_vars) == self.dim ** 2)

        # Solution constraint
        # the reason why we also need the variables for the solution is that we need the state in input to have
        # a feasible solution for it to be considerate an intermediate state of the problem
        for j in range(self.dim):
            for i in range(self.dim):
                pos_state = self.state[j, i]
                pos_sol = self.solution[j, i]
                for z in range(self.dim):
                    # the solution must be found starting from the values in the current state
                    self.model.add(self.model.if_then(self.model.sum(pos_state) == 1, pos_sol[z] == pos_state[z]))

        # Action constraint
        # the action must be the argmax of Y_vars
        action_value = self.model.max(self.Y_vars)
        for i in range(len(self.Y_vars)):
            self.model.add(self.model.if_then((self.action == i), (self.Y_vars[i] == action_value)))

        # Constraint to retrieve the next state given the action
        n = 0
        for j in range(self.dim):
            for i in range(self.dim):
                for z in range(self.dim):
                    if z == 0:
                        a1, a2 = 1, 2
                    elif z == 1:
                        a1, a2 = -1, 1
                    else:
                        a1, a2 = -1, -2

                    # set to 1 the bit corresponding to the action
                    self.model.add(self.model.if_then((self.action == n), (self.nextX_vars[n] == 1)))
                    # set to 0 the other 2 bits that are in the same position on the grid of the one that correspond
                    # to the action
                    # without this constraint, all the next state, where the agent change a number that was already
                    # filled in the input state, would be discard because of the one-hot encoding constraint
                    self.model.add(self.model.if_then(self.model.logical_or(self.action == n+a1, self.action == n+a2) == 1,
                                                      self.nextX_vars[n] == 0))
                    # if the bit is not in the same position on the grid of the one corresponding to the action
                    # then set it equal to the corresponding one on the current state
                    self.model.add(self.model.if_then(self.model.logical_and(self.action != n, self.action != n+a1, self.action != n+a2)==1,
                                                      (self.nextX_vars[n] == self.X_vars[n])))
                    n += 1

        # Unsafety constraint for the next state
        # it is unsafe when it is infeasible or when the agent change a number in the grid
        # that was already filled in the state in input
        for i in range(self.dim):
            for j in range(self.dim):
                next_pos = self.next_state[i, j]
                pos = self.state[i, j]
                ii = []
                jj = []
                constraint = []
                for z in range(self.dim):
                    if z != j:
                        jj.append(z)
                    if z != i:
                        ii.append(z)
                for z in range(self.dim - 1):
                    # if pos is the new added value, constrain it to be equal to another value in the same row or column
                    constraint.append(self.model.logical_and(self.next_state[i, jj[z], 0] == next_pos[0],
                                                             self.next_state[i, jj[z], 1] == next_pos[1],
                                                             self.next_state[i, jj[z], 2] == next_pos[2]))
                    constraint.append(self.model.logical_and(self.next_state[ii[z], j, 0] == next_pos[0],
                                                             self.next_state[ii[z], j, 1] == next_pos[1],
                                                             self.next_state[ii[z], j, 2] == next_pos[2]))

                # it constrains the next state to be infeasible just if an empty position is filled
                self.model.add(self.model.if_then((self.model.sum(pos) != self.model.sum(next_pos)),
                                                  self.model.logical_or(constraint[0], constraint[1],
                                                                        constraint[2], constraint[3]) == 1))

    # generate all possible solution with cpx.populate_solution_pool()
    def generate_solutions(self, mdl):
        cpx = mdl.get_cplex()
        cpx.parameters.mip.pool.intensity.set(4)  # enumerate ALL solutions
        cpx.parameters.mip.limits.populate.set(5000)  # solutions limit

        try:
            cpx.populate_solution_pool()
        except:
            print("Exception raised during populate")
            return []
        numsol = cpx.solution.pool.get_num()
        print(numsol)
        nb_vars = mdl.number_of_variables
        sol_pool = []
        for i in range(numsol):

            x_i = cpx.solution.pool.get_values(i)
            assert len(x_i) == nb_vars
            sol = mdl.new_solution()
            for k in range(nb_vars):
                vk = mdl.get_var_by_index(k)
                sol.add_var_value(vk, x_i[k])
            sol_pool.append(sol)
        return sol_pool

    def get_solutions(self):
        # Solutions
        print('=== Starting the solution process')
        start_time = time.time()
        sol = self.cpx.solve(self.model, 2000)

        if sol is None:
            print('=== NO SOLUTION FOUND')
        else:
            print('=== SOLUTION DATA')
            print('Solution time: {:.3f} (sec)'.format(sol['time']))
            print('Solver status: {}'.format(sol['status']))
            print('Cost: {}'.format(sol['obj']))

        # visualize all unique solutions
        if sol['status'] != "infeasible":
            pool = self.generate_solutions(self.model)
            full_sol = []
            feasible_sol = []

            for s, sol in enumerate(pool, start=1):
                sol_df = sol.as_df().replace(-0, 0)
                # print(sol_df.to_string())
                action = sol_df.iloc[108]['value']
                state = tuple(sol_df[:108:4]["value"])
                solution = tuple(sol_df[1:108:4]["value"])
                next_state = tuple(sol_df.loc[3:108:4]["value"])

                full_sol.append([state, next_state, action])
                feasible_sol.append([solution])

            df = pd.DataFrame(full_sol)
            df_feasible = pd.DataFrame(feasible_sol)
            df_feasible.columns = ['feasible_solution']
            df.columns = ['state', 'next_state', 'action']
            df = df.drop_duplicates()
            index = df.index.values.tolist()
            df_feasible = df_feasible.iloc[index].reset_index()
            df_feasible = df_feasible['feasible_solution']
            df = df.reset_index()

            states, next_states, actions = df['state'], df['next_state'], df['action']

            for state, next_state, action, feasible_solution, i in zip(states, next_states, actions, df_feasible, range(len(actions))):
                state_matrix = np.array(state).reshape((self.dim, self.dim, self.dim))
                next_state_matrix = np.array(next_state).reshape((self.dim, self.dim, self.dim))
                feasible_sol_matrix = np.array(feasible_solution).reshape((self.dim, self.dim, self.dim))

                print("Solution number #{}".format(i + 1))
                print("State")
                print(np.argmax(state_matrix, axis=2) + np.sum(state_matrix, axis=2))
                print("Next_state")
                print(np.argmax(next_state_matrix, axis=2) + np.sum(next_state_matrix, axis=2))
                print("Action: {}".format(action))
                print("Feasible solution")
                print(np.argmax(feasible_sol_matrix, axis=2) + np.sum(feasible_sol_matrix, axis=2))
                print("\n")

            print("Accuracy: {:.3f}%".format(100 - ((i + 1) / 5824 * 100)))

        last_time = time.time()
        t = last_time - start_time
        print("The solution took {:.3f} seconds".format(t))

    def solve(self):
        self.cpx = cplex_backend.CplexBackend()
        self.model = self.cpx.new_model()
        knet = self.load_keras_net()
        self.net = self.convert_keras_net(knet)

        self.set_variables()

        # encode net
        encode(self.cpx, self.net, self.model, self.X_vars, self.Y_vars, 'net_econding')

        self.set_constraints()
        self.get_solutions()


if __name__ == '__main__':
    dim = 3
    savepath = 'models/3x3_1layer'

    model = EMLmodel(savepath, dim)
    model.solve()

