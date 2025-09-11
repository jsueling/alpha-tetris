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

import numpy as np
import torch

from alpha_tetris.search.mcts import MCDecisionNodeAsync, ACTION_SPACE, C_PUCT
from alpha_tetris.models.reward_predictor import RewardPredictor

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)

C_FEATURE = 0.25  # Hyperparameter modulating the effect of feature scores on action selection
FEATURE_TEMP = 10 # Temperature to squash the feature scores using tanh

class UnsupervisedFeaturesMCDNA(MCDecisionNodeAsync):
    """
    This class implements a Monte Carlo Decision Node, augmented by biasing action selection
    search with features that were pre-extracted in an unsupervised manner.
    """

    # Class variable that holds the reward predictor instance. The same instance is
    # used across all nodes, saving time and memory.
    reward_predictor = RewardPredictor()

    def __init__(self, *args, **kwargs):
        super(UnsupervisedFeaturesMCDNA, self).__init__(*args, **kwargs)

        # Computed at evaluation
        self.feature_scores = None

    def get_best_action_by_puct(self) -> 'int':
        """
        Overrides the PUCT calculation to include the unsupervised feature scores.
        The feature score biases the search towards actions that are predicted to
        lead to higher discounted returns.
        """

        parent_visit_count = self.total_visit_count
        q_value_sums = self.q_value_sums[self.available_actions]
        visit_counts = self.visit_counts[self.available_actions]
        prior_probabilities = self.prior_probabilities[self.available_actions]
        feature_scores = self.feature_scores[self.available_actions]
        expected_q_value_estimates = np.zeros_like(q_value_sums, dtype=np.float32)

        np.divide(
            q_value_sums,
            visit_counts,
            out=expected_q_value_estimates,
            where=visit_counts > 0
        )

        # Modified PUCT: Exploit, explore and feature bias terms. The feature bias term is a
        # value function estimate for the state, conditioned on a compressed representation
        # of its most prominent features, learned by a modified Beta-VAE.

        puct_values = (
            expected_q_value_estimates +
            C_PUCT * prior_probabilities * np.sqrt(parent_visit_count) / (visit_counts + 1) +
            C_FEATURE * feature_scores
        )

        max_puct_value = float("-inf")
        max_puct_action_indices = []
        for i, puct_value in enumerate(puct_values):
            if puct_value > max_puct_value:
                max_puct_value = puct_value
                max_puct_action_indices = [i]
            elif puct_value == max_puct_value:
                max_puct_action_indices.append(i)

        return self.available_actions[np.random.choice(max_puct_action_indices)]

    async def evaluate(self, *args, **kwargs):
        """
        Override the evaluate method to include unsupervised feature biasing.
        This method uses the reward predictor to compute feature scores for each
        available action based on the predicted discounted return.
        """

        q_value = await super(UnsupervisedFeaturesMCDNA, self).evaluate(*args, **kwargs)

        if self.is_terminal:
            return q_value

        action_feature_scores = self.reward_predictor.calculate_feature_scores(
            self.available_actions,
            self.env
        )

        # Squash feature scores in the range [-1, 1] using tanh with selected
        # temperature hyperparameter
        normalised_feature_scores = np.tanh(action_feature_scores / FEATURE_TEMP)

        self.feature_scores = np.zeros(ACTION_SPACE, dtype=np.float32)
        self.feature_scores[self.available_actions] = normalised_feature_scores

        return q_value

class KnownFeatureMCDNA(MCDecisionNodeAsync):
    """
    This class implements a Monte Carlo Decision Node augmented with known feature biasing.
    """

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
