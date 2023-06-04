"""
    Implementation of RL agents and algorithms.
    Curiosity model is explained in the following paper: https://arxiv.org/pdf/1705.05363.pdf
    In the original version the model is composed of two nets: forward and inverse
    The inverse model is not implemented here since it's not useful for this problem
"""

import numpy as np
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from envs import PLSEnv
from tqdm import tqdm


class PLSAgent:

    def __init__(self, dim):
        self.env = PLSEnv(dim=dim)
        self.dim = dim
        self.num_inputs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.max_steps_per_episode = dim ** 2 + 1

    def build_curiosity_model(self):
        """
        Curiosity model: take in input current state and action and predict the next state
        """
        inputs1 = Input(shape=(self.num_inputs,), name='input_layer1')
        inputs2 = Input(shape=(1,), name='input_layer2')
        merged = Concatenate(axis=1)([inputs1, inputs2])
        dense = Dense(units=8, input_dim=2, activation="relu", name=f'hidden_0')(merged)
        out = Dense(self.num_inputs, name='output')(dense)

        return Model(inputs=[inputs1, inputs2], outputs=out, name='curiosity')

    def build_actor_critic(self):
        """
        Actor Critic model
        """
        inputs = Input(shape=(self.num_inputs,), name='input_layer')
        common = inputs

        for idx, units in enumerate(self.hidden_units):
            common = Dense(units=units, activation="relu", name=f'common_body_layer_{idx}')(common)

        actor = Dense(self.num_actions, activation="softmax", name='policy')(common)
        critic = Dense(1, name='critic')(common)

        return Model(inputs=inputs, outputs=[actor, critic], name='actor_critic')

    def generate_episode(self, episode):
        """
        Generate a full episode
        :return: float; last extrinsic reward
        :return: int; last timestep
        """
        state = self.env.reset()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)


        for timestep in range(1, self.max_steps_per_episode):
            # self.env.render() # Adding this line would show the attempts of the agent in a pop up window.

            # Predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = self.model(state)
            self.critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
            self.action_probs_history.append(tf.math.log(action_probs[0, action]))

            act = tf.convert_to_tensor(action)
            act = tf.expand_dims(act, 0)

            # Apply the sampled action in our environment
            state, reward, done, _ = self.env.step(action, episode, self.sparse_reward)
            state = state.astype(float)
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)


            # Curiosity model is not used from the start:
            # 1- At the beginning we already explore without the need of it (actually it decreases performance)
            # 2- If we use it from the start, it will learn too fast to predict the next state and will soon become unuseful
            if self.start_curiosity:
                # Predict next state with curiosity model
                pred_state = self.curiosity_model([state, act])
                pred_state = tf.convert_to_tensor(pred_state)
                pred_state = tf.expand_dims(pred_state, 0)

                # Compute intrinsic reward as a function of the loss of the curiosity model
                # the idea is: if the loss is high, the next state is not well known (not been there many times)
                # so we have a higher reward (increase exploration for that state)
                intrinsic_reward = (self.mse_loss(tf.expand_dims(pred_state, 0), tf.expand_dims(state, 0)) ** 2) * self.eta
                self.curiosity_losses.append(self.mse_loss(tf.expand_dims(pred_state, 0), tf.expand_dims(state, 0)))
                # Final reward is the sum of extrinsic (classic reward) and intrinsic reward
                final_reward = reward + intrinsic_reward
            else:
                final_reward = reward

            self.rewards_history.append(final_reward)
            self.episode_reward += final_reward

            if done:
                break

        return reward, timestep

    def HDF5_format(self):
        """
        Convert the final actor model in HDF5 format
        """
        actor_model = Sequential()

        for layer in self.model.layers:
            if layer.name == "input_layer" or layer.name.startswith("common_body_layer") or layer.name == "policy":
                actor_model.add(layer)

        actor_model.summary()
        return actor_model

    def save_model(self, savepath, model):
        """
        Save the final model
        :param savepath: string; name of the path where to save the model
        :param model: model to save
        """
        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            model.save(savepath)


    def train(self,
              num_episodes,
              hidden_units,
              seed=42,
              gamma1=1.0,
              gamma2=0.1,
              eta=1.0,
              savepath=None,
              curiosity=True,
              sparse_reward=False,
              verbose=True):
        """
        Train the RL agent.
        :param num_episodes: int; number of training episodes.
        :param hidden_units: list of int; the size of the list is the number of hidden layers and the values correspond to
                                          the number of units.
        :param seed: int; seed to ensure reproducible results.
        :param gamma1: float; discount factor for solved episodes.
        :param gamma2: float; discount factor for failed episodes.
        :param eta: scaling factor for intrinsic reward
        :param savepath: string; if not None, the model will be saved here.
        :param curiosity: boolean; True if you want to use the curiosity model
        :param sparse_reward: boolean; True if you want to use sparse reward
        :param verbose: boolean; True if you want to print all the loss values each 1000 episodes
        :return:
        """
        self.hidden_units = hidden_units
        self.gamma1= gamma1
        self.gamma2 = gamma2
        self.eta = eta
        self.sparse_reward = sparse_reward


        #self.env.seed(seed)
        #self.env.reset(seed)
        eps = np.finfo(np.float32).eps.item()

        # Keep track of the history
        history_dict = dict()
        history_dict['Reward'] = list()
        history_dict['Number of steps'] = list()
        history_dict['Actor loss'] = list()
        history_dict['Critic loss'] = list()
        history_dict['Curiosity loss'] = list()

        # Create Keras models
        self.model = self.build_actor_critic()
        self.model.summary()

        self.curiosity_model = self.build_curiosity_model()
        self.curiosity_model.summary()

        # Define optimizer and loss function
        self.optimizer = Adam()
        self.mse_loss = MeanSquaredError()

        # Keep track of history
        self.action_probs_history = list()
        self.critic_value_history = list()
        self.rewards_history = list()
        running_reward = 0
        episode_count = 0
        running_num_steps = 0
        self.start_curiosity = False

        # Run the training routine
        for episode in tqdm(range(num_episodes)):
            self.episode_reward = 0
            self.curiosity_losses = list()

            with tf.GradientTape(persistent=True) as tape:
                # generate a full episode
                reward, timestep = self.generate_episode(episode)

                # Update running reward and number of steps to check condition for solving
                running_reward = 0.05 * self.episode_reward + (1 - 0.05) * running_reward
                running_num_steps = 0.05 * timestep + (1 - 0.05) * running_num_steps

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = list()
                discounted_sum = 0
                if reward > 0:
                    for r in self.rewards_history[::-1]:
                        discounted_sum = r + gamma1 * discounted_sum
                        returns.insert(0, discounted_sum)
                else:
                    for r in self.rewards_history[::-1]:
                        discounted_sum = r + gamma2 * discounted_sum
                        returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action_probs_history, self.critic_value_history, returns)

                actor_losses = list()
                critic_losses = list()

                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        self.mse_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )
                    print(sum(actor_losses), sum(critic_losses))
                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                if self.start_curiosity:
                    curiosity_loss_value = sum(self.curiosity_losses)

            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.start_curiosity:
                grads2 = tape.gradient(curiosity_loss_value, self.curiosity_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads2, self.curiosity_model.trainable_variables))
            del tape

            mean_actor_loss = np.mean(actor_losses)
            mean_critic_loss = np.mean(critic_losses)

            if self.start_curiosity:
                mean_curiosity_loss = np.mean(self.curiosity_losses)
            else:
                mean_curiosity_loss = 0

            # Clear the loss and reward history
            self.action_probs_history.clear()
            self.critic_value_history.clear()
            self.rewards_history.clear()

            # Update the history
            history_dict['Reward'].append(running_reward)
            history_dict['Number of steps'].append(running_num_steps)
            history_dict['Actor loss'].append(mean_actor_loss)
            history_dict['Critic loss'].append(mean_critic_loss)
            history_dict['Curiosity loss'].append(mean_curiosity_loss)

            # Log details
            episode_count += 1
            if verbose:
                if episode_count % 100 == 0:
                    template = "[Epoch: {}] - Running reward: {:.2f} | Running # steps: {:.2f}"
                    template = template + " | Actor loss: {:.2f} | Critic loss: {:.2f} | Curiosity loss: {:.2f}"

                    print(template.format(episode_count,
                                          running_reward,
                                          running_num_steps,
                                          mean_actor_loss,
                                          mean_critic_loss,
                                          mean_curiosity_loss))

            # If.. then start using the curiosity model. (self.env.dim**2 * 4.5/10) could be considered a hyperparameter
            # it was decided to set it to this value after some experiments. The curiosity model is started to be used
            # after the main model learn to solve the first half (4.5/10) of the problem (the easiest part to solve)
            if curiosity:
                if self.start_curiosity is False and running_num_steps > (self.env.dim**2 * 0.45):
                    self.start_curiosity = True

        print('Solved during training: {}/{}'.format(self.env.sol, num_episodes))

        actor_model = self.HDF5_format()
        self.final_model = actor_model
        self.save_model(savepath, actor_model)

        return history_dict

    def test(self, num_episodes, savepath=None, verbose=True):
        """
        The test the RL agent saved in 'savepath'.
        :param num_episodes: int; number of training episodes.
        :param savepath: string; path where the model is saved. If it is None it will look for a model in self.final_model.
        :return:
        """
        if savepath == None:
            try:
                model = self.final_model
            except:
                print("The path to the model is no defined")
        else:
            model = tf.keras.models.load_model(savepath)
        sol = 0
        for idx in range(num_episodes):
            done = False
            s_t = self.env.reset()
            while not done:
                s_t = np.expand_dims(s_t, axis=0)
                action_probs = model(s_t)
                action_probs = action_probs.numpy()
                a_t = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
                s_t, r_t, done, info = self.env.step(a_t)
                if verbose:
                    self.env.render()
                    print()

            if done:
                if info['Solved']:
                    sol += 1
                    if verbose:
                        print('Solved!')
                        print()
                        print('-' * 50 + '\n')
                else:
                    if verbose:
                        print('Failed...')
                        print()
                        print('-' * 50 + '\n')


        print('Solved during test: {}/{}'.format(sol, num_episodes))



########################################################################################################################


if __name__ == '__main__':

    agent = PLSAgent(3)

    agent.train(num_episodes=10000,
          gamma1=1,
          gamma2=0.1,
          hidden_units=[32],
          eta=2,
          savepath='models/prova')

    #agent.test(num_episodes=100, savepath='models/3x3_1layer')

