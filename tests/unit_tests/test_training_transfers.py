"""Pinned rollout ownership and gradient logging must preserve training values."""

import contextlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pufferlib.pufferl import PuffeRL
from pufferlib.vector import Multiprocessing


@pytest.fixture
def shared_vector():
    # Isolate registration lifecycle from process startup and simulator configuration.
    vector = Multiprocessing.__new__(Multiprocessing)
    vector.processes = [Mock()]
    vector.buf = {"observations": np.arange(32, dtype=np.float32).reshape(2, 4, 4)}
    vector.zero_copy = True
    vector._driver_env_open = False
    vector._observation_registration = None
    return vector


def test_observation_registration_is_once_and_released_on_own_device(shared_vector, monkeypatch):
    runtime = Mock()
    runtime.cudaHostRegister.return_value = 0
    runtime.cudaHostUnregister.return_value = 0
    device_context = Mock(side_effect=lambda device: contextlib.nullcontext())
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    observations = shared_vector.buf["observations"]

    shared_vector.pin_observations()
    shared_vector.pin_observations()
    runtime.cudaHostRegister.assert_called_once_with(observations.ctypes.data, observations.nbytes, 0)
    synchronize.assert_not_called()

    shared_vector.close()
    shared_vector.close()
    device_context.assert_called_once_with(3)
    synchronize.assert_called_once()
    runtime.cudaHostUnregister.assert_called_once_with(observations.ctypes.data)
    assert shared_vector._observation_registration is None


def test_registration_failure_does_not_unregister_unowned_memory(shared_vector, monkeypatch):
    runtime = Mock()
    runtime.cudaHostRegister.return_value = 2
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        shared_vector.pin_observations()
    shared_vector.close()
    runtime.cudaHostUnregister.assert_not_called()


def test_copying_vector_does_not_register_unused_shared_buffer(shared_vector, monkeypatch):
    shared_vector.zero_copy = False
    runtime = Mock()
    monkeypatch.setattr(torch.cuda, "cudart", runtime)
    shared_vector.pin_observations()
    shared_vector.close()
    runtime.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for host registration")
def test_registered_observation_views_transfer_exactly_and_unpin(shared_vector):
    device = torch.cuda.current_device()
    shared_vector.pin_observations()
    view = torch.as_tensor(shared_vector.buf["observations"][1])
    try:
        assert view.is_pinned()
        copied = view.to(device=device, non_blocking=True)
        torch.testing.assert_close(copied.cpu(), view, rtol=0, atol=0)
    finally:
        shared_vector.close()
    assert not view.is_pinned()


class RejectScalarReadback(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("Gradient clipping read a scalar back to the CPU")
        return function(*args, **(kwargs or {}))


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_gradient_norms_stay_on_device_without_changing_updates(separate, device):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.config = {"max_grad_norm": 0.5}
    trainer.policy = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device)),
            torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device)),
        ]
    )
    trainer.actor_params = [trainer.policy[0]]
    trainer.critic_params = [trainer.policy[1]]
    trainer.separate_grad_clip = separate
    reference = [torch.nn.Parameter(parameter.detach().clone()) for parameter in trainer.policy]
    optimizer = torch.optim.AdamW(trainer.policy.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference, lr=0.01)
    groups = [[reference[0]], [reference[1]]] if separate else [reference]
    keys = ["actor_grad_norm", "critic_grad_norm"] if separate else ["grad_norm"]

    for magnitude in (1.0, 0.01, 10.0):
        for parameter, other in zip(trainer.policy, reference):
            gradient = torch.tensor([3.0, 4.0], device=device) * magnitude
            parameter.grad = gradient.clone()
            other.grad = gradient.clone()
        expected_norms = [torch.nn.utils.clip_grad_norm_(group, 0.5).item() for group in groups]
        losses = {}
        with RejectScalarReadback():
            trainer._clip_gradients(losses)
        optimizer.step()
        reference_optimizer.step()
        for key, expected_norm in zip(keys, expected_norms):
            assert isinstance(losses[key], torch.Tensor)
            assert losses[key].device.type == device
            assert not losses[key].requires_grad
            assert losses[key].item() == expected_norm
        for parameter, other in zip(trainer.policy, reference):
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
