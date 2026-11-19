import inspect
import itertools
import os
import numpy as np
from multiprocessing import get_all_start_methods, get_context

from opengigaflow import MAX_C_SEED, VecEnv, set_buffers
import opengigaflow.spaces

VECTOR_OPTIONS = ("num_workers", "batch_size", "observation_transport_dtype", "pin_workers")
RESET = 0
STEP = 1
SEND = 2
RECV = 3
WORKER_POLL_TIMEOUT_SECONDS = 5.0


def spawn_child_seeds(seed, count):
    if seed is None:
        return [None] * count
    return [int(child.generate_state(1)[0]) & MAX_C_SEED for child in np.random.SeedSequence(seed).spawn(count)]


def recv_precheck(vecenv):
    if vecenv.flag != RECV:
        raise opengigaflow.APIUsageError("Call reset before stepping")

    vecenv.flag = SEND


def _action_space_mismatch_message(space, actions):
    parts = [
        f"Actions do not match action space: actions shape={actions.shape} dtype={actions.dtype},"
        f" space shape={getattr(space, 'shape', None)} dtype={getattr(space, 'dtype', None)}"
    ]
    if np.issubdtype(actions.dtype, np.floating):
        nonfinite_count = int((~np.isfinite(actions)).sum())
        if nonfinite_count:
            parts.append(f"{nonfinite_count} non-finite values")
        low, high = getattr(space, "low", None), getattr(space, "high", None)
        if low is not None and actions.shape == space.shape:
            below, above = int((actions < low).sum()), int((actions > high).sum())
            if below or above:
                parts.append(
                    f"{below} below low / {above} above high,"
                    f" finite actions min={np.nanmin(actions)} max={np.nanmax(actions)}"
                )
    return "; ".join(parts)


def send_precheck(vecenv, actions):
    if vecenv.flag != SEND:
        raise opengigaflow.APIUsageError("Call (async) reset + recv before sending")

    actions = np.asarray(actions)
    if not vecenv.initialized:
        vecenv.initialized = True
        if not vecenv.action_space.contains(actions):
            raise opengigaflow.APIUsageError(_action_space_mismatch_message(vecenv.action_space, actions))

    vecenv.flag = RECV
    return actions


def reset(vecenv, seed=42):
    vecenv.async_reset(seed)
    obs, _, _, _, infos, _, _ = vecenv.recv()
    return obs, infos


def step(vecenv, actions):
    actions = np.asarray(actions)
    vecenv.send(actions)
    obs, rewards, terminals, truncations, infos, _, _ = vecenv.recv()
    return obs, rewards, terminals, truncations, infos


def make_config_env(creator, args, kwargs):
    if "config_only" in inspect.signature(creator).parameters:
        return creator(*args, **kwargs, config_only=True)
    return creator(*args, **kwargs)


class Serial:
    reset = reset
    step = step

    def __init__(self, env_creators, env_args, env_kwargs, num_envs, buf=None, seed=0, **kwargs):
        driver_env = make_config_env(env_creators[0], env_args[0], env_kwargs[0])
        num_agents_per_env = driver_env.num_agents
        self.agents_per_batch = num_agents_per_env * num_envs
        self.num_agents = self.agents_per_batch

        self.single_observation_space = driver_env.single_observation_space
        self.single_action_space = driver_env.single_action_space
        self.action_space = opengigaflow.spaces.joint_space(self.single_action_space, self.agents_per_batch)

        driver_env.close()

        set_buffers(self, buf)

        self.envs = []
        ptr = 0
        env_seeds = spawn_child_seeds(seed, num_envs)
        for i in range(num_envs):
            end = ptr + num_agents_per_env
            buf_i = dict(
                observations=self.observations[ptr:end],
                rewards=self.rewards[ptr:end],
                terminals=self.terminals[ptr:end],
                truncations=self.truncations[ptr:end],
                masks=self.masks[ptr:end],
                actions=self.actions[ptr:end],
            )
            ptr = end
            env = env_creators[i](*env_args[i], buf=buf_i, seed=env_seeds[i], **env_kwargs[i])
            self.envs.append(env)

        self.driver_env = self.envs[0]
        check_envs(self.envs, self.driver_env)
        self.agents_per_env = [env.num_agents for env in self.envs]
        assert sum(self.agents_per_env) == self.agents_per_batch
        self.agent_ids = np.arange(self.num_agents)
        self.initialized = False
        self.flag = RESET

    def async_reset(self, seed=None):
        self.flag = RECV
        self.infos = []
        for env, env_seed in zip(self.envs, spawn_child_seeds(seed, len(self.envs))):
            _, env_infos = env.reset(seed=env_seed)
            self.infos.extend(env_infos)

    def send(self, actions):
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)

        actions = send_precheck(self, actions)
        self.infos = []
        agent_start = 0
        for env, env_agent_count in zip(self.envs, self.agents_per_env):
            _, _, _, _, env_infos = env.step(actions[agent_start : agent_start + env_agent_count])
            self.infos.extend(env_infos)
            agent_start += env_agent_count

    def recv(self):
        recv_precheck(self)
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.infos,
            self.agent_ids,
            self.masks,
        )

    def close(self):
        for env in self.envs:
            env.close()


THREAD_SIBLINGS_PATH = "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
CACHE_DOMAIN_PATH = "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"


def read_cpu_cores(cpus):
    """Physical cores among `cpus` in CPU order, each as (L3 cache domain, hardware threads)."""
    cores = {}
    for cpu in sorted(cpus):
        with open(THREAD_SIBLINGS_PATH.format(cpu=cpu)) as siblings_file:
            core_key = siblings_file.read().strip()
        cache_domain = ""
        if os.path.exists(CACHE_DOMAIN_PATH.format(cpu=cpu)):
            with open(CACHE_DOMAIN_PATH.format(cpu=cpu)) as cache_file:
                cache_domain = cache_file.read().strip()
        cores.setdefault(core_key, (cache_domain, []))[1].append(cpu)
    return list(cores.values())


def plan_worker_cpus(cores, worker_count):
    """One hardware thread per worker, siblings last, cores round-robin over cache domains: buffers spread evenly."""
    domains = {}
    for cache_domain, core_threads in cores:
        domains.setdefault(cache_domain, []).append(core_threads)
    layers = itertools.zip_longest(*domains.values())
    interleaved_cores = [core_threads for layer in layers for core_threads in layer if core_threads is not None]
    sibling_layers = itertools.zip_longest(*interleaved_cores)
    hardware_threads = [cpu for layer in sibling_layers for cpu in layer if cpu is not None]
    return [hardware_threads[worker_idx % len(hardware_threads)] for worker_idx in range(worker_count)]


def cache_domain_shares(worker_cache_domains):
    """Per worker, the [start, end) fraction of a resource its cache domain owns, in proportion to the domain's workers."""
    worker_count = len(worker_cache_domains)
    domain_sizes = {}
    for cache_domain in worker_cache_domains:
        domain_sizes[cache_domain] = domain_sizes.get(cache_domain, 0) + 1
    domain_shares = {}
    workers_before = 0
    for cache_domain, domain_size in domain_sizes.items():
        domain_shares[cache_domain] = (workers_before / worker_count, (workers_before + domain_size) / worker_count)
        workers_before += domain_size
    return [domain_shares[cache_domain] for cache_domain in worker_cache_domains]


def _worker_process(
    env_creator,
    env_args,
    env_kwargs,
    obs_shape,
    obs_dtype,
    transport_dtype,
    atn_shape,
    atn_dtype,
    num_agents,
    num_workers,
    worker_idx,
    send_pipe,
    recv_pipe,
    shm,
    seed,
    worker_cpu,
):
    if worker_cpu is not None:
        os.sched_setaffinity(0, {worker_cpu})
    shape = (num_workers, num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=shm["actions"])[worker_idx]
    all_observations = np.ndarray((*shape, *obs_shape), dtype=transport_dtype, buffer=shm["observations"])
    shared_observations = all_observations[worker_idx]
    compress_observations = transport_dtype != obs_dtype
    env_observations = shared_observations
    if compress_observations:
        import torch

        torch.set_num_threads(1)
        env_observations = np.zeros(shared_observations.shape, dtype=obs_dtype)
        # torch's float16 cast is vectorized; numpy's costs 1 ms per 1024-agent step
        transport_tensor = torch.from_numpy(shared_observations)
        env_tensor = torch.from_numpy(env_observations)
    buf = dict(
        observations=env_observations,
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm["rewards"])[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm["terminals"])[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm["truncateds"])[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm["masks"])[worker_idx],
        actions=atn_arr,
    )
    buf["masks"][:] = True

    env = env_creator(*env_args, **env_kwargs, buf=buf, seed=seed)
    while True:
        command = recv_pipe.recv()
        if command == RESET:
            _, infos = env.reset(seed=recv_pipe.recv())
        elif command == STEP:
            _, _, _, _, infos = env.step(atn_arr)

        if compress_observations:
            transport_tensor.copy_(env_tensor)
        send_pipe.send(infos)


class Multiprocessing:
    reset = reset
    step = step

    def __init__(
        self,
        env_creators,
        env_args,
        env_kwargs,
        num_envs,
        num_workers=None,
        batch_size=None,
        seed=0,
        observation_transport_dtype=None,
        pin_workers=False,
    ):
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs
        if num_workers != num_envs:
            raise opengigaflow.APIUsageError(
                f"num_workers ({num_workers}) must equal num_envs ({num_envs}): each worker process runs one env"
            )
        if num_envs % batch_size != 0:
            raise opengigaflow.APIUsageError("num_envs must be divisible by batch_size")

        self.workers_per_batch = batch_size
        self.num_workers = num_workers

        self.driver_env = driver_env = make_config_env(env_creators[0], env_args[0], env_kwargs[0])
        self.num_agents = num_agents = driver_env.num_agents * num_envs
        self.agents_per_batch = driver_env.num_agents * batch_size
        agents_per_worker = driver_env.num_agents
        obs_space = driver_env.single_observation_space
        obs_shape = obs_space.shape
        obs_dtype = obs_space.dtype
        transport_dtype = np.dtype(observation_transport_dtype or obs_dtype)
        if transport_dtype != obs_dtype and not (transport_dtype == np.float16 and obs_dtype == np.float32):
            raise opengigaflow.APIUsageError(f"Cannot transport {obs_dtype} observations as {transport_dtype}")
        self.observation_transport_dtype = transport_dtype
        # ctypes has no float16, so float16 observations live in a uint16 array
        obs_ctype = "H" if transport_dtype == np.float16 else np.ctypeslib.as_ctypes_type(obs_dtype)
        atn_space = driver_env.single_action_space
        atn_shape = atn_space.shape
        atn_dtype = atn_space.dtype

        atn_ctype = np.ctypeslib.as_ctypes_type(atn_dtype)

        self.single_observation_space = driver_env.single_observation_space
        self.single_action_space = driver_env.single_action_space
        self.action_space = opengigaflow.spaces.joint_space(self.single_action_space, self.agents_per_batch)
        self.agent_ids = np.arange(num_agents).reshape(num_workers, agents_per_worker)

        preload_shared_resources = getattr(driver_env, "preload_shared_resources", None)
        self._driver_env_open = False
        if "fork" in get_all_start_methods() and callable(preload_shared_resources):
            self._driver_env_open = bool(preload_shared_resources())
        process_context = get_context("fork") if self._driver_env_open else get_context()
        if not self._driver_env_open:
            driver_env.close()

        # RawArray, not SharedMemory: SharedMemory's resource tracker spams warnings and breaks forked workers
        self.shm = dict(
            observations=process_context.RawArray(obs_ctype, num_agents * int(np.prod(obs_shape))),
            actions=process_context.RawArray(atn_ctype, num_agents * int(np.prod(atn_shape))),
            rewards=process_context.RawArray("f", num_agents),
            terminals=process_context.RawArray("b", num_agents),
            truncateds=process_context.RawArray("b", num_agents),
            masks=process_context.RawArray("b", num_agents),
        )
        shape = (num_workers, agents_per_worker)
        self.obs_batch_shape = (self.agents_per_batch, *obs_shape)
        self.atn_batch_shape = (self.workers_per_batch, agents_per_worker, *atn_shape)
        self.actions = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=self.shm["actions"])
        self.buf = dict(
            observations=np.ndarray((*shape, *obs_shape), dtype=transport_dtype, buffer=self.shm["observations"]),
            rewards=np.ndarray(shape, dtype=np.float32, buffer=self.shm["rewards"]),
            terminals=np.ndarray(shape, dtype=bool, buffer=self.shm["terminals"]),
            truncations=np.ndarray(shape, dtype=bool, buffer=self.shm["truncateds"]),
            masks=np.ndarray(shape, dtype=bool, buffer=self.shm["masks"]),
        )

        self.send_pipes, w_recv_pipes = zip(*[process_context.Pipe() for _ in range(num_workers)])
        w_send_pipes, self.recv_pipes = zip(*[process_context.Pipe() for _ in range(num_workers)])

        worker_seeds = spawn_child_seeds(seed, num_workers)
        self.worker_cpus = [None] * num_workers
        self.worker_cache_shares = [None] * num_workers
        self.driver_cpus = None
        self._released_driver_cpus = None
        if pin_workers:
            cores = read_cpu_cores(os.sched_getaffinity(0))
            if len(cores) > 1:
                # a driver thread that shares a core with a worker slows that worker's whole buffer
                _, driver_core_threads = cores.pop()
                self.driver_cpus = set(driver_core_threads)
            self.worker_cpus = plan_worker_cpus(cores, num_workers)
            cpu_cache_domains = {cpu: cache_domain for cache_domain, core_threads in cores for cpu in core_threads}
            self.worker_cache_shares = cache_domain_shares([cpu_cache_domains[cpu] for cpu in self.worker_cpus])

        self.processes = []
        for worker_idx in range(num_workers):
            worker_env_kwargs = env_kwargs[worker_idx]
            takes_cache_share = "cache_domain_share" in inspect.signature(env_creators[worker_idx]).parameters
            if takes_cache_share and self.worker_cache_shares[worker_idx] is not None:
                # envs that keep per-env data hot (maps) partition it so that one cache domain's workers share it
                worker_env_kwargs = dict(worker_env_kwargs, cache_domain_share=self.worker_cache_shares[worker_idx])
            process = process_context.Process(
                target=_worker_process,
                args=(
                    env_creators[worker_idx],
                    env_args[worker_idx],
                    worker_env_kwargs,
                    obs_shape,
                    obs_dtype,
                    transport_dtype,
                    atn_shape,
                    atn_dtype,
                    agents_per_worker,
                    num_workers,
                    worker_idx,
                    w_send_pipes[worker_idx],
                    w_recv_pipes[worker_idx],
                    self.shm,
                    worker_seeds[worker_idx],
                    self.worker_cpus[worker_idx],
                ),
                daemon=True,  # a driver that fails before close() must exit instead of waiting on its workers
            )
            process.start()
            self.processes.append(process)

        self.flag = RESET
        self.initialized = False
        self._observation_registration = None
        self.pending_result = np.zeros(num_workers, dtype=bool)

    def _check_worker_alive(self, worker):
        process = self.processes[worker]
        if not process.is_alive():
            raise RuntimeError(f"Vectorization worker {worker} exited with code {process.exitcode}")

    def _recv_worker_result(self, worker):
        pipe = self.recv_pipes[worker]
        while not pipe.poll(WORKER_POLL_TIMEOUT_SECONDS):
            self._check_worker_alive(worker)

        self.pending_result[worker] = False
        infos = pipe.recv()
        if infos:
            self.infos[worker] = infos

    def recv(self):
        recv_precheck(self)
        # whole worker blocks in send order keep each batch aligned with the experience buffer
        batch_start = self.next_batch_start
        batch_end = batch_start + self.workers_per_batch
        self.batch_workers = range(batch_start, batch_end)
        self.batch_slice = slice(batch_start, batch_end)
        self.next_batch_start = batch_end % self.num_workers
        for worker in self.batch_workers:
            self._recv_worker_result(worker)

        buf = self.buf
        observations = buf["observations"][self.batch_slice].reshape(self.obs_batch_shape)
        rewards = buf["rewards"][self.batch_slice].ravel()
        terminals = buf["terminals"][self.batch_slice].ravel()
        truncations = buf["truncations"][self.batch_slice].ravel()

        infos = []
        for worker in self.batch_workers:
            if self.infos[worker]:
                infos.extend(self.infos[worker])
                self.infos[worker] = []

        agent_ids = self.agent_ids[self.batch_slice].ravel()
        masks = buf["masks"][self.batch_slice].ravel()
        return observations, rewards, terminals, truncations, infos, agent_ids, masks

    def send(self, actions):
        actions = send_precheck(self, actions).reshape(self.atn_batch_shape)

        self.actions[self.batch_slice] = actions
        for worker in self.batch_workers:
            self.send_pipes[worker].send(STEP)
        self.pending_result[self.batch_slice] = True

    def async_reset(self, seed=0):
        for worker in np.flatnonzero(self.pending_result):
            self._recv_worker_result(worker)

        self.flag = RECV
        self.next_batch_start = 0
        self.infos = [[] for _ in range(self.num_workers)]
        for worker, worker_seed in enumerate(spawn_child_seeds(seed, self.num_workers)):
            self.send_pipes[worker].send(RESET)
            self.send_pipes[worker].send(worker_seed)
        self.pending_result[:] = True

    def bind_driver_thread(self):
        """Move the calling thread onto the core kept free of pinned workers; release_driver_thread undoes it."""
        if self.driver_cpus is None:
            return
        self._released_driver_cpus = os.sched_getaffinity(0)
        os.sched_setaffinity(0, self.driver_cpus)

    def release_driver_thread(self):
        if self._released_driver_cpus is None:
            return
        os.sched_setaffinity(0, self._released_driver_cpus)
        self._released_driver_cpus = None

    def pin_observations(self):
        """Register stable shared observations in the trainer's current CUDA context."""
        if self.processes is None:
            raise opengigaflow.APIUsageError("Cannot pin observations after the vector environment is closed")
        if self._observation_registration is not None:
            return

        import torch

        observations = self.buf["observations"]
        device = torch.cuda.current_device()
        error = torch.cuda.cudart().cudaHostRegister(observations.ctypes.data, observations.nbytes, 0)
        if int(error) != 0:
            raise RuntimeError(f"cudaHostRegister failed for observations ({observations.nbytes} bytes): {error}")
        self._observation_registration = (observations, device)

    def close(self):
        if self.processes is None:
            return
        for p in self.processes:
            p.terminate()
        if self._observation_registration is not None:
            import torch

            observations, device = self._observation_registration
            # Finish any asynchronous reads before releasing this rank's registration.
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                error = torch.cuda.cudart().cudaHostUnregister(observations.ctypes.data)
            if int(error) != 0:
                raise RuntimeError(f"cudaHostUnregister failed for observations: {error}")
            self._observation_registration = None
        if self._driver_env_open:
            self.driver_env.close()
            self._driver_env_open = False
        self.processes = None


VECTOR_BACKENDS = {"Serial": Serial, "Multiprocessing": Multiprocessing}


def make(env_creator_or_creators, env_args=None, env_kwargs=None, *, backend, num_envs=1, seed=0, **kwargs):
    if num_envs < 1:
        raise opengigaflow.APIUsageError("num_envs must be at least 1")
    if num_envs != int(num_envs):
        raise opengigaflow.APIUsageError("num_envs must be an integer")
    if backend not in VECTOR_BACKENDS:
        raise opengigaflow.APIUsageError(f"Invalid backend: {backend}")

    if kwargs.get("num_workers") == "auto":
        kwargs["num_workers"] = num_envs
    if kwargs.get("batch_size") == "auto":
        kwargs["batch_size"] = max(num_envs // 2, 1)
    if "num_workers" in kwargs:
        envs_per_worker = num_envs / kwargs["num_workers"]
        if envs_per_worker != int(envs_per_worker):
            raise opengigaflow.APIUsageError("num_envs must be divisible by num_workers")
        batch_size = kwargs.get("batch_size")
        if batch_size is not None and batch_size % envs_per_worker != 0:
            raise opengigaflow.APIUsageError("batch_size must be divisible by (num_envs / num_workers)")

    if env_args is None:
        env_args = []

    if env_kwargs is None:
        env_kwargs = {}

    if not isinstance(env_creator_or_creators, (list, tuple)):
        env_creators = [env_creator_or_creators] * num_envs
        env_args = [env_args] * num_envs
        env_kwargs = [env_kwargs] * num_envs
    else:
        env_creators = env_creator_or_creators

    if len(env_creators) != num_envs:
        raise opengigaflow.APIUsageError("env_creators must be a list of length num_envs")
    if len(env_args) != num_envs:
        raise opengigaflow.APIUsageError("env_args must be a list of length num_envs")
    if len(env_kwargs) != num_envs:
        raise opengigaflow.APIUsageError("env_kwargs must be a list of length num_envs")

    for i in range(num_envs):
        if not callable(env_creators[i]):
            raise opengigaflow.APIUsageError("env_creators must be a list of callables")
        if not isinstance(env_args[i], (list, tuple)):
            raise opengigaflow.APIUsageError("env_args must be a list of lists or tuples")
        if not isinstance(env_kwargs[i], dict):
            raise opengigaflow.APIUsageError("env_kwargs must be a list of dictionaries")

    # Keeps batch size consistent when debugging with Serial backend
    if backend == "Serial" and kwargs.get("batch_size") is not None:
        num_envs = kwargs["batch_size"]

    for k in kwargs:
        if k not in VECTOR_OPTIONS:
            raise opengigaflow.APIUsageError(f"Invalid argument: {k}")

    return VECTOR_BACKENDS[backend](env_creators, env_args, env_kwargs, num_envs, seed=seed, **kwargs)


def check_envs(envs, driver):
    if not isinstance(driver, VecEnv):
        raise opengigaflow.APIUsageError("env_creator must be VecEnv")

    driver_obs = driver.single_observation_space
    driver_atn = driver.single_action_space
    for env in envs:
        if not isinstance(env, VecEnv):
            raise opengigaflow.APIUsageError("env_creators must be VecEnv")
        obs_space = env.single_observation_space
        if obs_space != driver_obs:
            raise opengigaflow.APIUsageError(f"\n{obs_space}\n{driver_obs} obs space mismatch")
        atn_space = env.single_action_space
        if atn_space != driver_atn:
            raise opengigaflow.APIUsageError(f"\n{atn_space}\n{driver_atn} atn space mismatch")
