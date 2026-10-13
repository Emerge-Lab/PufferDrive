"""save_checkpoint keeps the newest model plus every model whose epoch is a multiple of checkpoint_keep_interval."""

import os
import sys
import tempfile
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.pufferl import PuffeRL

ENV_NAME = "puffer_drive_spline_werling"
SAVED_EPOCHS = (50, 100, 150, 200, 250)


def _trainer(data_dir, keep_interval):
    policy = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    return types.SimpleNamespace(
        config={"data_dir": data_dir, "env": ENV_NAME, "checkpoint_keep_interval": keep_interval},
        logger=types.SimpleNamespace(run_id="keep_test"),
        uncompiled_policy=policy,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
        global_step=0,
        epoch=0,
        last_stats={},
        best_score=float("inf"),
        ema_max=0.0,
    )


def _kept_epochs(keep_interval):
    with tempfile.TemporaryDirectory() as data_dir:
        trainer = _trainer(data_dir, keep_interval)
        for epoch in SAVED_EPOCHS:
            trainer.epoch = epoch
            PuffeRL.save_checkpoint(trainer)
        names = sorted(os.listdir(os.path.join(data_dir, "models")))
        assert os.path.exists(os.path.join(data_dir, "trainer_state.pt"))
    return [int(name[: -len(".pt")].rsplit("_", 1)[1]) for name in names]


class TestCheckpointKeep(unittest.TestCase):
    def test_zero_keeps_only_the_newest_model(self):
        self.assertEqual(_kept_epochs(0), [250])

    def test_keep_interval_also_keeps_its_multiples(self):
        self.assertEqual(_kept_epochs(100), [100, 200, 250])


if __name__ == "__main__":
    unittest.main()
