"""
This file contains modifications of the base MCTS algorithm for Tetris found in mcts.py:
1. Biasing with features that were extracted in an unsupervised manner (UnsupervisedFeaturesMCDNA).
   Features were extracted from game states of an expert Tetris controller using a Beta-VAE. The
   Beta-VAE was then modified to predict discounted return which is used here to bias action
   selection search.
2. Biasing with known features of Tetris (KnownFeatureMCDNA). This is a simple and naive
   heuristic that prunes all available actions that create more than the minimum number of
   possible holes among the grids resulting from each available action.
"""

import torch

from mcts import MCDecisionNodeAsync

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

class UnsupervisedFeaturesMCDNA(MCDecisionNodeAsync):

    def __init__(self, *args, **kwargs):
        super(UnsupervisedFeaturesMCDNA, self).__init__(*args, **kwargs)


class KnownFeatureMCDNA(MCDecisionNodeAsync):

    def __init__(self, *args, **kwargs):
        super(KnownFeatureMCDNA, self).__init__(*args, **kwargs)

    async def evaluate(self, *args, **kwargs):
        """
        Override the evaluate method to include known feature biasing.
        This method naively prunes all available actions that create more
        than the minimum number of possible holes among the grids resulting
        from each available action.
        """
        q_value = await super(KnownFeatureMCDNA, self).evaluate(*args, **kwargs)
        if self.is_terminal:
            return q_value
        self.available_actions = self.env.prune_hole_creation(self.available_actions)
        return q_value
