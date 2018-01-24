# ================================================
# Modified from the work of Arthur Juliani:
#       Simple Reinforcement Learning with Tensorflow Part 8: Asynchronus Advantage Actor-Critic (A3C)
#       https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
#
#       Implementation of Asynchronous Methods for Deep Reinforcement Learning
#       Algorithm details can be found here:
#           https://arxiv.org/pdf/1602.01783.pdf
#
# Author: Mark Edmonds
# =================================================

import os
import threading
import multiprocessing
import sys
import numpy as np
import tensorflow as tf
import gym

from worker import Worker
from ac_network import AC_Network

from gym_lock.session_manager import SessionManager
from gym_lock.settings_trial import PARAMS, IDX_TO_PARAMS
from gym_lock.settings_scenario import select_scenario
from gym_lock.space_manager import ObservationSpace, ActionSpace

# ==========================
#   Training Parameters
# ==========================
RANDOM_SEED = 1234
# Load previously trained model
LOAD_MODEL = False
# Test and visualise a trained model
TEST_MODEL = False
# Learning rate
LEARNING_RATE = 0.0001
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99
# LSTM cell unit size
CELL_UNITS = 256

def main(_):
    global master_network
    global global_episodes

    reward_mode = None
    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS['CE3-CE4']
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS['CC4']
    else:
        setting = sys.argv[1]
        params = PARAMS[IDX_TO_PARAMS[int(setting)-1]]
        print('training_scenario: {}, testing_scenario: {}'.format(params['train_scenario_name'], params['test_scenario_name']))
        reward_mode = sys.argv[2]

    use_physics = False
    num_training_iters = 100

    # RL specific settings
    params['data_dir'] = '../../OpenLockA3CResults/subjects/'
    params['train_attempt_limit'] = 300
    params['test_attempt_limit'] = 300
    params['use_physics'] = False
    params['num_training_iters'] = 100
    params['reward_mode'] = reward_mode

    scenario = select_scenario(params['train_scenario_name'], use_physics=use_physics)

    ENV_NAME = 'arm_lock-v0'

    env = gym.make(ENV_NAME)

    # create session/trial/experiment manager
    manager = SessionManager(env, params, human=False)
    manager.update_scenario(scenario)
    trial_selected = manager.run_trial_common_setup(scenario_name=params['train_scenario_name'],
                                                    action_limit=params['train_action_limit'],
                                                    attempt_limit=params['train_attempt_limit'])

    env.observation_space = ObservationSpace(len(scenario.levers))
    MODEL_DIR = manager.writer.subject_path + '/models'
    MONITOR_DIR = manager.writer.subject_path + '/monitor'

    STATE_DIM = env.observation_space.shape
    ACTION_DIM = len(env.action_space)

    # delete temporary env
    env.close()

    tf.reset_default_graph()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with tf.device("/cpu:0"):
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, CELL_UNITS, 'global', None)  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads

        # For testing and visualisation we only need one worker
        if TEST_MODEL:
            num_workers = 1

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(name=i,
                                  s_size=STATE_DIM,
                                  a_size=ACTION_DIM,
                                  trainer=trainer,
                                  model_path=MODEL_DIR,
                                  global_episodes=global_episodes,
                                  env_name=ENV_NAME,
                                  seed=RANDOM_SEED,
                                  test=TEST_MODEL,
                                  cell_units=CELL_UNITS,
                                  params=params))
        saver = tf.train.Saver(max_to_keep=5)

        # Gym monitor
        if not TEST_MODEL:
            env = workers[0].get_env()
            env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=False, force=True)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if LOAD_MODEL or TEST_MODEL:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        if TEST_MODEL:
            env = workers[0].get_env()
            env = gym.wrappers.Monitor(env, MONITOR_DIR, force=True)
            workers[0].work(GAMMA, sess, coord, saver)
        else:
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate thread.
            print 'Launching workers...'
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(GAMMA, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)

if __name__ == '__main__':
    tf.app.run()