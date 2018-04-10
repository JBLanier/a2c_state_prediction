import gym
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model, Sequential
from keras import backend as K
import keras
import numpy as np
from envs.subproc_vec_env import SubprocVecEnv
from lr_decay import LearningRateDecay
import tensorflow as tf


class StatePredictor:
    def __init__(self, observation_space_shape, num_actions):
        model = Sequential()

        print("input shape: {}".format((observation_space_shape[0] + num_actions,)))

        model.add(Dense(input_shape=(observation_space_shape[0] + num_actions,), units=80, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(40, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(observation_space_shape[0] + 1, activation='linear', kernel_initializer='he_normal'))

        model.compile(loss='mse', optimizer='adam')

        self.model = model
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions

    def _format_inputs(self, current_observations, actions):
        actions_one_hot = np.zeros((actions.shape[0], self.num_actions))
        for i, action in enumerate(actions):
            actions_one_hot[i, action] = 1
        return np.column_stack((current_observations, actions_one_hot))

    def predict(self, current_observations, actions):
        return self.model.predict_on_batch(self._format_inputs(current_observations, actions))

    def train(self, current_observations, actions, rewards, next_observations):

        inputs = self._format_inputs(current_observations, actions)
        targets = np.column_stack((next_observations, rewards))
        # print("c_obs: {}".format(current_observations))
        # print("actions: {}".format(actions))
        # print("rewards: {}".format(rewards))
        # print("next_observations: {}".format(next_observations))
        # print("inputs: {}".format(inputs))
        # print("targets: {}".format(targets))
        return self.model.train_on_batch(x=inputs, y=targets)


class A2CAgent:
    def __init__(self, observation_space_shape, num_actions):

        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions

        self.policy_entropy_regularization_factor = 0.01
        self.value_loss_coefficient = 1

        observations = Input(shape=self.observation_space_shape, dtype='float32', name='observations')
        h1 = Dense(units=40, activation=keras.activations.relu, name='fc1')(observations)
        h2 = Dense(units=80, activation=keras.activations.relu, name='fc2')(h1)
        h3 = Dense(units=40, activation=keras.activations.relu, name='fc3')(h2)

        policy_probabilities_outputs = Dense(units=self.num_actions,
                                             activation=keras.activations.softmax,
                                             name='policy_output')(h3)

        values_outputs = Dense(units=1,
                               activation=None,
                               name='value_output')(h3)

        self.model = Model(inputs=observations, outputs=[policy_probabilities_outputs, values_outputs])

        advantages = K.placeholder(shape=(None, self.num_actions))
        value_targets = K.placeholder(shape=None)

        policy_gradient_loss = K.categorical_crossentropy(target=advantages, output=policy_probabilities_outputs)

        policy_entropy = K.categorical_crossentropy(target=policy_probabilities_outputs,
                                                    output=policy_probabilities_outputs)

        value_loss = keras.losses.mean_squared_error(y_true=value_targets, y_pred=values_outputs)

        total_loss = policy_gradient_loss - self.policy_entropy_regularization_factor * policy_entropy + value_loss * self.value_loss_coefficient

        # self.optimizer = keras.optimizers.adam(lr=0.001)
        # updates = self.optimizer.get_updates(loss=total_loss, params=self.model.trainable_weights)

        params = self.model.trainable_weights

        grads = tf.gradients(total_loss, params)
        # max_grad_norm = 0.5
        # if max_grad_norm is not None:
        #     grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        _train = trainer.apply_gradients(grads)

        self.train_fn = K.function(
            inputs=[advantages, value_targets, observations],
            outputs=[policy_probabilities_outputs, values_outputs,
                     policy_gradient_loss, policy_entropy, value_loss, total_loss],
            updates=[_train]
        )

    def step(self, observation):
        action_probs, predicted_value = self.model.predict_on_batch(x={'observations': observation})
        return action_probs, predicted_value

    def train(self, observations, values, actions, discounted_rewards):
        advantages = np.zeros((values.shape[0], self.num_actions))
        for i, (action, value, discounted_reward) in enumerate(zip(actions, values, discounted_rewards)):
            advantages[i, action] = discounted_reward - value
        # K.set_value(self.optimizer.lr, lr)
        self.train_fn([advantages, discounted_rewards, observations])



class Trainer:

    def __init__(self):
        self.env_id = "CartPole-v1"
        self.num_env = 2
        self.seed = 42
        self.n_steps = 3
        self.num_iterations = 1000000
        self.discount_factor = 0.99
        self.dones = [False for _ in range(self.num_env)]
        # self.lr_decay = LearningRateDecay(v=0.0001, nvalues=self.num_iterations*self.num_env*self.n_steps, lr_decay_method='linear')

        def make_env(rank):
            def _thunk():
                env = gym.make(self.env_id)
                env.seed(self.seed + rank)
                # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                return env
            return _thunk

        self.env = SubprocVecEnv([make_env(i) for i in range(self.num_env)])
        self.observations = self.env.reset()

        self.train_observation_shape = (self.num_env * self.n_steps,) + self.env.observation_space.shape

        self.agent = A2CAgent(observation_space_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n)
        self.state_predictor = StatePredictor(observation_space_shape=self.env.observation_space.shape, num_actions=self.env.action_space.n)

        for iteration in range(self.num_iterations):

            mb_observations, mb_discounted_rewards, mb_actions, mb_values, mb_next_obs, mb_rewards = self.__run_steps()
            # print("mb_observations: {}".format(mb_observations))
            # print("mb_discounted_rewards: {}".format(mb_discounted_rewards))
            # print("mb_actions: {}".format(mb_actions))
            # print("mb_values: {}".format(mb_values))

            self.agent.train(mb_observations, mb_values, mb_actions, mb_discounted_rewards)

            loss = self.state_predictor.train(mb_observations, mb_actions, mb_rewards, mb_next_obs)
            if iteration % 50 == 0:
                print("state pred loss: {}".format(loss))

    def __run_steps(self):

        mb_observations, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        for n in range(self.n_steps):
            # self.env.render()

            action_probabilities, values = self.agent.step(self.observations)

            actions = []
            for p in action_probabilities:
                actions.append(np.random.choice(self.env.action_space.n, p=p))

            mb_observations.append(self.observations)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            new_observations, rewards, dones, _ = self.env.step(actions)

            mb_rewards.append(rewards)

            self.dones = dones
            self.observations = new_observations
        mb_dones.append(self.dones)
        mb_observations.append(self.observations)
        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        # print("As array: {}".format(np.asarray(mb_observations, dtype=np.float32)))
        mb_next_obs = np.asarray(mb_observations[1:], dtype=np.float32).swapaxes(1, 0).reshape(self.train_observation_shape)


        mb_obs = np.asarray(mb_observations[:-1], dtype=np.float32).swapaxes(1, 0).reshape(self.train_observation_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_discounted_rewards = np.zeros(mb_rewards.shape, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        _, last_values = self.agent.step(self.observations)

        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.discount_factor)[:-1]
            else:
                rewards = self.__discount_with_dones(rewards, dones, self.discount_factor)
            mb_discounted_rewards[n] = rewards

        # Instead of (num_envs, time_steps). Make them num_envs*time_steps.
        mb_discounted_rewards = mb_discounted_rewards.flatten()
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        return mb_obs, mb_discounted_rewards, mb_actions, mb_values, mb_next_obs, mb_rewards

    @staticmethod
    def __discount_with_dones(rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

if __name__ == '__main__':
    trainer = Trainer()
