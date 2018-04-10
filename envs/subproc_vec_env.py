import numpy as np
from multiprocessing import Process, Pipe


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    step_counter = -1
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            step_counter += 1
            ob, reward, done, info = env.step(data)
            total_info = info.copy()  # Very important for passing by value instead of reference
            if done:
                if step_counter != 500:
                    reward = -10
                # print("Done after {} steps".format(step_counter))
                step_counter = 0
                ob = env.reset()
            remote.send((ob, reward, done, total_info))
        elif cmd == 'reset':
            step_counter = 0
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            step_counter = 0
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class SubprocVecEnv:
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    @property
    def num_envs(self):
        return len(self.remotes)


