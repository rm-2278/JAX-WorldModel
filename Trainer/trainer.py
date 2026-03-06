import os
import abc
import jax.random as random
from flax.training import train_state, checkpoints


class Trainer(abc.ABC):
    """Abstract base trainer. Subclasses must implement init_model and _train_epoch."""

    def __init__(self, seed: int = 42, ckpt_prefix: str = 'model'):
        self.seed = seed
        self.ckpt_prefix = ckpt_prefix
        self.init_key = random.key(seed)
        # Repository-level checkpoint directory (two levels up from Trainer/)
        self.ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoint'))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.init_model()

    @abc.abstractmethod
    def init_model(self):
        """Create self.model and self.state. Called at end of __init__."""
        pass

    @abc.abstractmethod
    def _train_epoch(self, key, epoch: int, num_epochs: int):
        """Run one training epoch. Must return (key, loss)."""
        pass

    # ------------------------------------------------------------------
    # Training loop (template method)
    # ------------------------------------------------------------------

    def train(self, key, num_epochs: int = 10):
        for epoch in range(num_epochs):
            key, loss = self._train_epoch(key, epoch, num_epochs)
            self._on_epoch_end(epoch)

    def _on_epoch_end(self, epoch: int):
        """Called after every epoch. Override to add extra behaviour (e.g. visualisation)."""
        self.save_model(epoch)

    # ------------------------------------------------------------------
    # Checkpoint helpers (parameterised by ckpt_prefix)
    # ------------------------------------------------------------------

    def save_model(self, step: int):
        checkpoints.save_checkpoint(
            ckpt_dir=self.ckpt_dir, step=step,
            target=self.state.params, prefix=self.ckpt_prefix, overwrite=True,
        )

    def load_model(self):
        params = checkpoints.restore_checkpoint(
            ckpt_dir=self.ckpt_dir, target=self.state.params, prefix=self.ckpt_prefix,
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn, params=params, tx=self.state.tx,
        )

    def checkpoint_exists(self) -> bool:
        return checkpoints.latest_checkpoint(self.ckpt_dir, prefix=self.ckpt_prefix) is not None
