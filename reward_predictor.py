"""Reward predictor (modified Beta-VAE) for Tetris game states."""

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
import numpy as np

if TYPE_CHECKING:
    from tetris_env import Tetris

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else \
    ("mps" if torch.backends.mps.is_available() else "cpu")
)
LATENT_DIM = 8
GRID_HEIGHT = 20
GRID_WIDTH = 10

class TetrisDiscountedReturnVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states using a convolutional architecture.
    This model encodes the game state (grid) into a regularised latent space, and
    decodes it, to reconstruct the original input and predict the discounted return.
    """

    def __init__(
        self,
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        latent_dim=LATENT_DIM,
    ):

        super(TetrisDiscountedReturnVAE, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.latent_dim = latent_dim

        # Encoder
        encoder_channels = [32, 64, 128, 256]

        encoder_layers = []
        input_channels = 1
        for output_channels in encoder_channels:
            encoder_layers.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the flattened encoder output size dynamically after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.grid_height, self.grid_width)
            dummy_output = self.encoder(dummy_input)
            self.conv_output_size = int(np.prod(dummy_output.shape))
            self.conv_output_shape = dummy_output.shape[1:]

        # Latent space
        self.fc_mean = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

        # Decoder
        decoder_channels = [256, 128, 64]

        # Maps from latent space to the number of features of the first decoder layer
        self.fc_decode = nn.Linear(latent_dim, self.conv_output_size)

        decoder_layers = []
        input_channels = decoder_channels[0]
        for output_channels in decoder_channels:
            decoder_layers.append(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            )
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        # Final layer
        decoder_layers.append(
            nn.ConvTranspose2d(
                input_channels,
                1,
                kernel_size=(5, 3),
                stride=1,
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes the input into mean and variance vectors of the latent space vector z."""
        x = self.encoder(x)
        x = x.flatten(start_dim=1)  # Flatten the convolutional encoder output
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

    def reparameterise(self, z_mean, z_logvar):
        """Applies the reparameterisation trick to sample z from the latent space distribution."""
        std = torch.exp(0.5 * z_logvar) # σ = exp(0.5 × log(σ²)) = √(σ²)
        eps = torch.randn_like(std)
        return z_mean + eps * std # z = μ + σ × ε

    def decode(self, z: Tensor):
        """
        Decodes the latent space vector z back to the original input space.
        returns logits of the reconstructed grid
        """
        # Preprocess z for decoding - expand latent_dim to flattened conv_output_size
        z = self.fc_decode(z)
        # Reshape z to match the output shape of the last convolutional layer
        z = z.view(-1, *self.conv_output_shape)
        # Invert the convolutional layers to reconstruct the grid
        z = self.decoder(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """
        Predicts the discounted return for a given input grid. The model was
        trained to regularise the latent space and predict the discounted return
        of tetris grids as a distribution (mu, sigma).
        """
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterise(z_mean, z_logvar)
        discounted_rewards = self.reward_predictor(z)
        rewards_mu, _ = discounted_rewards[:, 0], discounted_rewards[:, 1]
        return rewards_mu

class RewardPredictor:
    """
    This class encapsulates the reward prediction model for Tetris game states.
    It loads a pretrained Beta-VAE model that predicts the discounted return for Tetris game states.
    """

    def __init__(self):

        pretrained_model_path = "./pretrained_models/tetris_discounted_return_vae.pth"

        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError("Pretrained model for TetrisDiscountedReturnVAE not found.")

        base_reward_predictor = TetrisDiscountedReturnVAE().to(DEVICE)

        rp_model, dr_mean, dr_std = self.load_discounted_return_model(
            base_reward_predictor,
            pretrained_model_path
        )

        # The model is used for inference only
        self.reward_predictor = rp_model.to(DEVICE).eval()
        if torch.cuda.is_available():
            self.reward_predictor = torch.compile(self.reward_predictor)
        # Used for unnormalising the predicted discounted return distribution
        # (mu 0, sigma 1) to the original distribution
        self.train_discounted_return_mean = dr_mean
        self.train_discounted_return_std = dr_std

        height_bin_statistics_path = "./pretrained_models/reward_prediction_height_bin_stats.npy"

        if not os.path.exists(height_bin_statistics_path):
            raise FileNotFoundError("Height bin stats for normalising"
                                    " predicted discounted return not found.")

        self.height_bin_stats = np.load(height_bin_statistics_path, allow_pickle=True).item()
        self.bins = np.array(list(range(0, 19, 2)) + [21])
        labels = [f"{self.bins[i]}-{self.bins[i+1]-1}" for i in range(len(self.bins)-1)]
        self.lookup_mu_mean = np.array([self.height_bin_stats[l]['mu_mean'] for l in labels])
        self.lookup_mu_std = np.array([self.height_bin_stats[l]['mu_std'] for l in labels])

    def calculate_feature_scores(self, available_actions, env: "Tetris"):
        """
        Calculates feature scores for the available actions based on the
        predicted discounted return. This method uses the reward predictor
        to compute feature scores for each available action.
        """

        grids = env.get_grids_after_available_actions(available_actions)

        grid_input = torch.from_numpy(np.expand_dims(grids, axis=1)).to(DEVICE)

        # The model was trained to output a prediction distribution (mu, sigma) of discounted
        # return that was normalised to (0, 1) for training stability
        with torch.no_grad():
            pred_normalised_reward_mu = \
                self.reward_predictor(grid_input).cpu().numpy()

        # Unnormalise the normalised predictions to the original distribution range
        pred_reward_mu = \
            pred_normalised_reward_mu * self.train_discounted_return_std \
                + self.train_discounted_return_mean

        # Normalise the predicted mu of the distribution by stack height (sigma is not used)
        height_normalising_mu_mean, height_normalising_mu_std = \
            self.get_height_normalising_stats(
                grids,
                self.bins,
                self.lookup_mu_mean,
                self.lookup_mu_std
            )

        height_normalised_reward_mu = \
            (pred_reward_mu - height_normalising_mu_mean) / height_normalising_mu_std

        return height_normalised_reward_mu

    def get_height_normalising_stats(
        self,
        grids: np.ndarray,
        bins: np.ndarray,
        lookup_mu_mean: np.ndarray,
        lookup_mu_std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For the given grids, fetches the mean and std of predicted mu discounted return,
        categorised and precomputed by stack height.
        Returns:
        - mu_mean: Mean of predicted mu discounted return for each grid's stack height bin
        - mu_std: Std of predicted mu discounted return for each grid's stack height bin
        """

        heights = self.get_stack_height(grids)

        bin_indices = np.digitize(heights, bins) - 1

        mu_mean = lookup_mu_mean[bin_indices]
        mu_std = lookup_mu_std[bin_indices]

        return mu_mean, mu_std

    def load_discounted_return_model(self, model: TetrisDiscountedReturnVAE, path):
        """
        Loads the model state dictionary and normalisation stats from the specified path.
        The normalisation stats are used to unnormalise the predicted discounted returns
        from distribution (0, 1) to the original distribution of the training data.
        Returns:
        - model: The TetrisDiscountedReturnVAE model with loaded state dict.
        - train_discounted_return_mean: Mean of the discounted return used for normalisation.
        - train_discounted_return_std: Std of the discounted return used for normalisation.
        """
        loaded_model_data = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(loaded_model_data['model_state_dict'])
        train_discounted_return_mean = loaded_model_data['discounted_return_mean']
        train_discounted_return_std = loaded_model_data['discounted_return_std']
        return model, train_discounted_return_mean, train_discounted_return_std

    def get_stack_height(self, grids: np.ndarray) -> np.ndarray:
        """
        Returns the highest filled cell in any column of the grid given grids of shape (B, H, W)
        """
        # Add a row of ones at the bottom in case the grid is empty
        # (argmax returns 0 => stack height 20)
        grids = np.concatenate((grids, np.ones_like(grids[:, :1, :])), axis=1)
        # Assigns 1 for each row if any cell is filled in it or 0 otherwise
        rows_with_any_filled = grids.any(axis=2).astype(np.float32) # B H
        # Argmax finds the row index of the highest filled cell
        highest_filled_row_indices = \
            rows_with_any_filled.argmax(axis=1) # B
        # Convert to height since rows are indexed moving down the column
        return GRID_HEIGHT - highest_filled_row_indices
