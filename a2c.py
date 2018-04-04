import gym
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model, Sequential
from keras import backend as K
import keras
import numpy as np


class TDStatePredictor:
    def __init__(self, observation_space_shape, sequence_length):
        batch_size = 1  # equals number of simultaneous environments?

        model = Sequential()

        model.add(LSTM(32, return_sequences=True, stateful=False,
                       batch_input_shape=(batch_size, sequence_length,) + observation_space_shape))
        model.add(LSTM(32, return_sequences=True, stateful=False))
        model.add(TimeDistributed(Dense(20, activation='relu', kernel_initializer='he_normal')))
        model.add(TimeDistributed(Dense(observation_space_shape[0], activation='linear')))

        model.compile(loss='mse', optimizer='rmsprop')

        self.model = model
        self.sequence_length = sequence_length
        self.observation_space_shape = observation_space_shape

    def _repeat_inputs(self, observations):
        observations = np.expand_dims(observations, axis=1)
        return np.repeat(observations, repeats=self.sequence_length, axis=1)

    def predict_future_observations(self, current_observation):
        # - this is one to many -

        # input for every time step of rnn is the initial observation repeated
        return self.model.predict_on_batch(self._repeat_inputs(current_observation))

    def train(self, current_observation, next_observation):
        y_target = self.predict_future_observations(next_observation)[:, :-1, :]
        y_target = np.insert(y_target, 0, next_observation, axis=1)

        return self.model.train_on_batch(x=self._repeat_inputs(current_observation), y=y_target)


class A2CAgent:
    def __init__(self, observation_space_shape, num_actions):

        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions

        self.policy_entropy_regularization_factor = 0.01
        self.value_loss_coefficient = 1
        self.discount_factor = 0.99

        observations = Input(shape=self.observation_space_shape, dtype='float32', name='observations')
        h1 = Dense(units=40, activation=keras.activations.relu, name='fc1')(observations)
        h2 = Dense(units=40, activation=keras.activations.relu, name='fc2')(h1)

        policy_probabilities_outputs = Dense(units=self.num_actions,
                                             activation=keras.activations.softmax,
                                             name='policy_output')(h2)

        values_outputs = Dense(units=1,
                               activation=None,
                               name='value_output')(h2)

        self.model = Model(inputs=observations, outputs=[policy_probabilities_outputs, values_outputs])

        advantages = K.placeholder(shape=(None, self.num_actions))
        value_targets = K.placeholder(shape=None)

        policy_gradient_loss = K.categorical_crossentropy(target=advantages, output=policy_probabilities_outputs)

        policy_entropy = K.categorical_crossentropy(target=policy_probabilities_outputs,
                                                    output=policy_probabilities_outputs)

        value_loss = keras.losses.mean_squared_error(y_true=value_targets, y_pred=values_outputs)

        total_loss = policy_gradient_loss - self.policy_entropy_regularization_factor * policy_entropy + value_loss * self.value_loss_coefficient

        optimizer = keras.optimizers.adam(lr=0.001)
        updates = optimizer.get_updates(loss=total_loss, params=self.model.trainable_weights)

        self.train_fn = K.function(
            inputs=[advantages, value_targets, observations],
            outputs=[policy_probabilities_outputs, values_outputs,
                     policy_gradient_loss, policy_entropy, value_loss, total_loss],
            updates=updates
        )

    def step(self, observation):
        action_probs, predicted_value = self.model.predict_on_batch(x={'observations': observation})
        return action_probs, predicted_value

    def train(self, observation, value, action, reward, next_observation, done):
        _, new_value = self.model.predict_on_batch(x={'observations': next_observation})

        if done:
            value_target = reward
        else:
            value_target = reward + (self.discount_factor * new_value)

        advantage = np.zeros((1, self.num_actions))
        advantage[0, action] = value_target - value

        self.train_fn([advantage, value_target, observation])


def main():
    # TODO: Implement multiple simultaneous environments and multi-step returns for A2C

    env = gym.make('CartPole-v1')
    agent = A2CAgent(observation_space_shape=env.observation_space.shape, num_actions=env.action_space.n)
    state_predictor = TDStatePredictor(observation_space_shape=env.observation_space.shape, sequence_length=5)

    for i_episode in range(20000):
        observation = env.reset()
        observation = np.expand_dims(observation, axis=0)

        for t in range(1000):
            # env.render()

            action_probabilities, value = agent.step(observation)
            action_choice = np.random.choice(env.action_space.n, p=action_probabilities[0])

            new_observation, reward, done, _ = env.step(action_choice)
            new_observation = np.expand_dims(new_observation, axis=0)

            agent.train(observation, value, action_choice, reward, new_observation, done)

            if i_episode > 400:
                # let A2C agent gain some experience first
                td_loss = state_predictor.train(current_observation=observation, next_observation=new_observation)
                if i_episode % 100 == 0 and t % 100 == 0:
                    print("state prediction loss: {}".format(td_loss))


            observation = new_observation
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                # if t + 1 == 500:
                #     print("Completed Cart Pole Task! ({} Episodes)".format(i_episode))
                break

    print("Done")
    env.close()


if __name__ == '__main__':
    main()
