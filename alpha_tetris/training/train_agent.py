"""Main training script for Deep MCTS agent."""

import multiprocessing as mp
import random
import numpy as np
import torch

from alpha_tetris.agents.deep_mcts_agent_async import DeepMCTSAgentAsync
from alpha_tetris.agents.deep_mcts_agent_ensemble import DeepMCTSAgentEnsemble

def train_agent(agent_type: str, checkpoint_name: str, seed: int):
    """Main function to set up and start training of the Deep MCTS agent."""

    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        # Benchmarks, then caches most efficient convolution algorithms
        # given the current configuration. Do not use if input sizes change frequently
        torch.backends.cudnn.benchmark = True
        # Enables use of fast TF32 Tensor Cores for matrix multiplications
        torch.set_float32_matmul_precision('high')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if agent_type == "async":
        agent = DeepMCTSAgentAsync(checkpoint_name=checkpoint_name)
        agent.train()
    elif agent_type == "ensemble":
        agent = None
        try:
            agent = DeepMCTSAgentEnsemble(checkpoint_name=checkpoint_name)
            agent.train()
        finally:
            # Clean up resources
            if agent is not None:
                agent.stop()
