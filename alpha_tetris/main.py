"""Main entry point for the program, parses command line arguments."""

import argparse

from alpha_tetris.training.train_agent import train_agent

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Deep MCTS agent to play Tetris.")

    parser.add_argument(
        "--agent_type",
        "-a",
        choices=["async", "ensemble"],
        required=True,
        help="Type of Deep MCTS agent to train: 'async' for DeepMCTSAgentAsync or "
             "'ensemble' for DeepMCTSAgentEnsemble."
    )

    parser.add_argument(
        "--checkpoint_name",
        "-c",
        required=True,
        type=str,
        help="Name of the checkpoint for loading and saving the state of training"
    )

    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    checkpoint_name = args.checkpoint_name + f"_{args.agent_type}_seed_{args.seed}"

    train_agent(
        agent_type=args.agent_type,
        checkpoint_name=checkpoint_name,
        seed=args.seed
    )
