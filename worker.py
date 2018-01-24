import tensorflow as tf
import scipy.signal
import numpy as np
import gym
from ac_network import AC_Network

from gym_lock.session_manager import SessionManager
from gym_lock.settings_scenario import select_scenario

# Size of mini batches to run training on
MINI_BATCH = 30
REWARD_FACTOR = 0.001

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Weighted random selection returns n_picks random indexes.
# the chance to pick the index i is give by the weight weights[i].
def weighted_pick(weights,n_picks):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t,np.random.rand(n_picks)*s)

# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Normalization of inputs and outputs
def norm(x, upper, lower=0.):
    return (x-lower)/max((upper-lower), 1e-12)

class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, env_name, seed, test, cell_units, params, testing_trial=False):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.is_test = test
        self.a_size = a_size
        self.params = params

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, cell_units, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.testing_trial = testing_trial
        if not self.testing_trial:
            self.scenario_name = params['train_scenario_name']
            self.attempt_limit = params['train_attempt_limit']
        else:
            self.scenario_name = params['test_scenario_name']
            self.attempt_limit = params['test_attempt_limit']

        self.scenario = select_scenario(self.scenario_name, params['use_physics'])
        self.env = gym.make(env_name)

        self.manager = SessionManager(self.env, params, human=False)
        self.manager.update_scenario(self.scenario)
        self.env.reward_mode = params['reward_mode']

        self.trial_count = 0
        self.env.seed(seed)

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist()+[r])*REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist()+[r])*REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            sess.run(self.update_local_ops)
            episode_buffer = []
            episode_mini_buffer = []
            episode_values = []
            episode_states = []
            episode_reward = 0
            episode_step_count = 0

            if not self.testing_trial:
                trial_selected = self.manager.run_trial_common_setup(self.params['train_scenario_name'], self.params['train_action_limit'], self.params['train_attempt_limit'], multithreaded=True)
            else:
                trial_selected = self.manager.run_trial_common_setup(self.params['test_scenario_name'], self.params['test_action_limit'], self.params['test_attempt_limit'], specified_trial='trial7', multithreaded=True)

            self.env.reset()
            while not coord.should_stop():

                # update trial if needed
                if self.env.attempt_count > self.attempt_limit or self.env.logger.cur_trial.success is True:
                    if not self.testing_trial:
                        trial_selected = self.manager.run_trial_common_setup(self.params['train_scenario_name'], self.params['train_action_limit'], self.params['train_attempt_limit'], multithreaded=True)
                    else:
                        trial_selected = self.manager.run_trial_common_setup(self.params['test_scenario_name'], self.params['test_action_limit'], self.params['test_attempt_limit'], specified_trial='trial7', multithreaded=True)
                    print('scenario_name: {}, trial_count: {}, trial_name: {}'.format(self.scenario_name, self.trial_count, trial_selected))
                    sess.run(self.update_local_ops)
                    episode_buffer = []
                    episode_mini_buffer = []
                    episode_values = []
                    episode_states = []
                    episode_reward = 0
                    episode_step_count = 0
                    self.trial_count += 1
                    self.env.reset()

                # Restart environment
                done = False
                state = self.env.reset()

                rnn_state = self.local_AC.state_init

                # Run an episode
                while not done:
                    episode_states.append(state)
                    if self.is_test:
                        self.env.render()

                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs: [state],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})

                    a0 = weighted_pick(a_dist[0], 1) # Use stochastic distribution sampling
                    if self.is_test:
                        a0 = np.argmax(a_dist[0]) # Use maximum when testing
                    a = np.zeros(self.a_size)
                    a[a0] = 1

                    next_state, reward, done, opt = self.env.step(np.argmax(a), multithreaded=False)

                    episode_reward += reward

                    episode_buffer.append([state, a, reward, next_state, done, v[0, 0]])
                    episode_mini_buffer.append([state, a, reward, next_state, done, v[0, 0]])

                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    if len(episode_mini_buffer) == MINI_BATCH and not self.is_test:
                        v1 = sess.run([self.local_AC.value],
                                      feed_dict={self.local_AC.inputs: [state],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_mini_buffer, sess, gamma, v1[0][0])
                        episode_mini_buffer = []

                    # Set previous state for next step
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if episode_count % 100 == 0 and not episode_count % 1000 == 0 and not self.is_test:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Scenario name', simple_value=str(self.env.scenario.name))
                    summary.value.add(tag='trial count', simple_value=str(self.trial_count))
                    summary.value.add(tag='trial name', simple_value=str(trial_selected))
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    if episode_count % 1000 == 0 and not self.is_test:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

                    print("| Reward: " + str(episode_reward), " | Episode", episode_count)
                    sess.run(self.increment) # Next global episode

                episode_count += 1
