"""
Tetris environment for reinforcement learning agents.
Adapted from: https://github.com/corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method
"""

import random
from typing import Optional

import numpy as np

ACTION_SPACE = 40 # Maximum number of placements: 4 rotations * 10 columns

class Tetromino:
    """Represents a Tetromino piece with its position, type, and rotation."""

    figures = [
        # Each figure is represented by a flattened 4x4 dimension
        # with a list of each 4 rotations it can take
        [[0, 4, 8, 12], [0, 1, 2, 3]],                            # I
        [[0, 1, 5, 6], [1, 4, 5, 8]],                             # Z
        [[4, 5, 1, 2], [0, 4, 5, 9]],                             # S
        [[1, 0, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [0, 1, 2, 6]], # J
        [[0, 1, 5, 9], [4, 0, 1, 2], [0, 4, 8, 9], [4, 5, 6, 2]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [0, 1, 2, 5], [0, 4, 8, 5]], # T
        [[0, 1, 4, 5]],                                           # O
    ]

    default_spawns = [
        # Default spawns are horizontally centered along the top row
        # format: column, rotation
        [3, 1],  # I
        [3, 0],  # Z
        [3, 0],  # S
        [3, 3],  # J
        [3, 1],  # L
        [3, 2],  # T
        [4, 0],  # O
    ]

    # Each Tetromino's position is represented by
    # The following 4x4 grid:

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15

    # A Tetromino's (x, y) coordinates reference the position
    # of where cell 0 in the 4x4 grid is located in the 10x20 grid.

    def __init__(self, tetromino_type: int):
        # Instantiated with no position or rotation
        self.x = None
        self.y = None
        self.type = tetromino_type # type of the Tetromino (0-6)
        self.rotation = None

    def spawn(self, x: int, rotation: int):
        """
        Sets the Tetromino's position to row 0, in the specified column (x)
        and rotation.
        """
        self.x = x
        self.y = 0
        self.rotation = rotation # rotation of the Tetromino (0-3)

    def despawn(self):
        """Despawn the Tetromino by resetting its position and rotation."""
        self.x = None
        self.y = None
        self.rotation = None

    def image(self):
        """Returns the current 4x4 image of the Tetromino based on its type and rotation."""
        return self.figures[self.type][self.rotation]

class Tetris:
    """
    Represents the Tetris game state,
    including the grid, current tetromino, score, and game state.
    """
    def __init__(
        self,
        height: int=20,
        width: int=10,
        piece_gen_scheme: str="uniform"
    ):

        self.tetromino = None
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.score = np.float32(0)
        self.done = False

        # scheme can be type "uniform" or "bag"
        self.piece_gen_scheme = piece_gen_scheme

        if self.piece_gen_scheme == "bag":
            # Initialise a bag of random Tetrominoes
            self.bag = list(range(len(Tetromino.figures)))
            random.shuffle(self.bag)

    def __repr__(self):
        grid_repr = []
        for row in range(self.height):
            grid_repr.append(
                "".join(str(np.int64(self.grid[row][col])) for col in range(self.width))
            )
        return "\n".join(grid_repr)

    def create_tetromino(self, tetromino_type: int) -> None:
        """Creates a new unspawned Tetromino of tetromino_type attached to this game state."""
        self.tetromino = Tetromino(tetromino_type)

    def get_current_tetromino_type(self) -> Optional[int]:
        """
        Returns the type of the current Tetromino (0-6).
        If no Tetromino is currently active, returns None.
        """
        if self.tetromino:
            return self.tetromino.type
        return None

    def generate_next_tetromino_type(self, is_first_tetromino=False) -> int:
        """
        Generates the next Tetromino type based on the randomisation scheme.
        For "uniform", it returns a random Tetromino type.
        For "bag", it returns a Tetromino type from a bag containing each
        Tetromino in a random order, refilling and reshuffling it when empty.
        """

        if self.piece_gen_scheme == "uniform":
            # Randomly select a Tetromino type uniformly
            return random.randint(0, 6)

        if self.piece_gen_scheme == "bag":

            # If this is the first Tetromino of the episode
            # or the bag is empty, refill and shuffle it
            if is_first_tetromino or not self.bag:
                self.bag = list(range(len(Tetromino.figures)))
                random.shuffle(self.bag)
            return self.bag.pop()

        raise ValueError(
            f"Invalid tetromino randomisation scheme: {self.piece_gen_scheme}"
        )

    def intersects(self):
        """
        Returns True if the current Tetromino placement is invalid or False otherwise
        (OOB and collision checks).
        """
        x, y = self.tetromino.x, self.tetromino.y
        for cell_index in self.tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            if ( # OOB checks and collision check
                tetromino_row < 0 or
                tetromino_row >= self.height or
                tetromino_col >= self.width or
                tetromino_col < 0 or
                self.grid[tetromino_row][tetromino_col] > 0
            ):
                return True
        return False

    def break_lines(self):
        """Break lines that are completely filled by Tetrominoes."""

        filled_lines = np.all(self.grid > 0, axis=1)
        # All filled lines now become broken lines
        broken_lines = np.count_nonzero(filled_lines)
        if broken_lines > 0:
            self.grid = np.vstack((
                # Add empty rows at the top to replace the broken lines
                np.zeros((broken_lines, self.width), dtype=np.float32),
                # Keep only rows that were not filled (maintains ordering)
                self.grid[~filled_lines]
            ))
            self.score = np.float32(self.score + broken_lines)

    def hard_drop(self, colour=1):
        """
        Move the current Tetromino directly down to the bottom of the grid.
        After the hard drop, the Tetromino is removed from the game state.
        """
        while not self.intersects():
            self.tetromino.y += 1
        self.tetromino.y -= 1
        self.freeze(colour)
        self.tetromino = None

    def freeze(self, colour):
        """Freeze the current Tetromino, it now becomes part of the grid."""
        x, y = self.tetromino.x, self.tetromino.y
        for cell_index in self.tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            self.grid[tetromino_row][tetromino_col] = colour
        self.break_lines()

    def path_exists_to_col(self, target_column):
        """
        Check if there is a valid path across the top row to the desired column
        for the Tetromino hard drop placement. This assumes that
        all shifts and rotations are possible during lock delay.
        """
        current_column = self.tetromino.x
        if target_column < current_column:
            for col in range(current_column, target_column - 1, -1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        elif target_column > current_column:
            for col in range(current_column, target_column + 1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        return True

    def get_legal_actions(self):
        """
        Returns a list of legal actions over the action space for the current Tetromino.
        Each action is represented as 2 digits in base 10:
        - The first digit is the rotation (0-3).
        - The second digit is the column (0-9).

        Legal actions here are defined using the simplified version of Tetris
        which assumes that the Tetromino can be spawned in any column.
        Redundant rotations are also masked to avoid unnecessary actions.

        Returns:
            np.array: A boolean array of shape (40,) where each index corresponds
            to a possible action (rotation * width + column).
            True indicates that the action is legal, False indicates it is illegal.
        """

        # All actions are initially illegal. Actions are represented
        # as indices in base max_columns as rotation * max_columns + column
        # where rotation is in [0, 3] and column is in [0, 9]
        legal_actions = np.zeros(ACTION_SPACE, dtype=bool)

        # Get the number of unique rotations for the current Tetromino
        # since Tetrominoes have symmetric rotations.
        unique_rotations = len(Tetromino.figures[self.get_current_tetromino_type()])

        for rotation in range(unique_rotations):
            for col in range(self.width):
                # Try to spawn the Tetromino in the specified column and rotation
                self.tetromino.spawn(x=col, rotation=rotation)
                if not self.intersects():
                    legal_actions[rotation * self.width + col] = True

        # Despawn the Tetromino after checking all legal actions
        # to ensure the state is clean for the next action
        self.tetromino.despawn()

        return legal_actions

    def step(self, action_idx: int, colour=1):
        """
        Perform a step in the game by applying the given action.

        Each action is represented as 2 digits in base 10:
        - The first digit is the rotation (0-3).
        - The second digit is the column (0-9).

        Returns:
        - done (bool): True if the game is over, False otherwise.
        """
        if action_idx < 0 or action_idx >= self.width * 4:
            return True

        rotation, col = divmod(action_idx, self.width)
        self.tetromino.spawn(x=col, rotation=rotation)

        if self.intersects():
            return True

        # Current Tetromino is placed and removed from the game state
        self.hard_drop(colour)

        return self.done

    def reset(self):
        """
        Reset the game state to the initial conditions.
        """
        self.grid.fill(0)
        self.score = np.float32(0)
        self.done = False
        # Generate the first Tetromino of the new episode
        self.create_tetromino(
            self.generate_next_tetromino_type(is_first_tetromino=True)
        )

    def copy(self):
        """Create a deep copy of the current game state."""
        new_env = Tetris(piece_gen_scheme=self.piece_gen_scheme)
        np.copyto(new_env.grid, self.grid)
        new_env.score = self.score
        new_env.done = self.done
        if self.tetromino:
            new_env.create_tetromino(self.get_current_tetromino_type())
        if self.piece_gen_scheme == "bag":
            new_env.bag = self.bag.copy()
        return new_env

    def get_state(self):
        """
        Returns a representation of the current game state used by the dual-headed neural network
        for selecting actions and evaluating the game state:
        - The first layer encodes the grid, where each cell is either 0 (empty) or 1 (filled).
        - Layers 1 to 7 are a one-hot encoding of the current Tetromino type
        (filled with 1s when active).
        """
        state = np.zeros((1 + len(Tetromino.figures), self.height, self.width), dtype=np.float32)
        np.copyto(state[0], self.grid)
        tetromino_type = self.get_current_tetromino_type()
        state[tetromino_type + 1, :, :] = 1.0
        return state

    def get_grid(self):
        """
        Returns the current grid of the Tetris game, used by the reward predictor neural network.
        """
        grid = np.zeros((1, self.height, self.width), dtype=np.float32)
        np.copyto(grid[0], self.grid)
        return grid

    def save_partial_state(self):
        """
        Returns a tuple containing the flattened representation of the grid
        and current Tetromino type.
        """
        partial_state = (self.grid.flatten(), self.get_current_tetromino_type())
        return partial_state

    def load_partial_state(self, partial_state):
        """
        Load a state from a flattened representation.
        The state is expected to be a tuple of (grid, tetromino_type).

        Note: Does not copy any other game state attributes like score or done.
        """
        flattened_grid, tetromino_type = partial_state
        self.grid = flattened_grid.reshape((self.height, self.width))
        self.create_tetromino(tetromino_type)

    def _intersects_stateless(self, grid, tetromino: Tetromino) -> bool:
        """Stateless version of intersects that operates on a given grid and Tetromino."""
        x, y = tetromino.x, tetromino.y
        for cell_index in tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            if ( # OOB checks and collision check
                tetromino_row < 0 or
                tetromino_row >= self.height or
                tetromino_col >= self.width or
                tetromino_col < 0 or
                grid[tetromino_row][tetromino_col] > 0
            ):
                return True
        return False

    def _break_lines_stateless(self, grid):
        """Stateless version of break_lines that operates on a given grid."""
        filled_lines = np.all(grid > 0, axis=1)
        broken_lines = np.count_nonzero(filled_lines)
        if broken_lines > 0:
            new_grid = np.vstack((
                np.zeros((broken_lines, self.width), dtype=np.float32),
                grid[~filled_lines]
            ))
            return new_grid
        return grid

    def count_holes(self, grid: np.ndarray) -> int:
        """Count the number of holes in the grid."""
        filled_mask = grid > 0
        cumulative_filled = np.cumsum(filled_mask, axis=0)
        # A hole is an empty cell below a filled cell
        holes = (~filled_mask) & (cumulative_filled > 0)
        return holes.sum()

    def prune_hole_creation(self, available_actions):
        """
        Simulates all available actions and returns the set of actions with minimum hole creation.
        Note: this can be negative if the action breaks lines that uncover holes.
        """

        tmp_grid = np.zeros_like(self.grid)
        tmp_tetromino = Tetromino(self.get_current_tetromino_type())

        holes_before = self.count_holes(tmp_grid)
        holes_created = np.full_like(available_actions, fill_value=float("inf"), dtype=np.float32)

        for i, action_idx in enumerate(available_actions):

            rotation, col = divmod(action_idx, self.width)
            tmp_tetromino.spawn(x=col, rotation=rotation)

            np.copyto(tmp_grid, self.grid)

            # Hard drop
            while not self._intersects_stateless(tmp_grid, tmp_tetromino):
                tmp_tetromino.y += 1
            tmp_tetromino.y -= 1

            # Freeze
            x, y = tmp_tetromino.x, tmp_tetromino.y
            for cell_index in tmp_tetromino.image():
                row, col = y + (cell_index // 4), x + (cell_index % 4)
                tmp_grid[row][col] = 1

            # Break lines
            final_grid = self._break_lines_stateless(tmp_grid)

            # Store holes created by this action
            holes_created[i] = self.count_holes(final_grid) - holes_before

        return available_actions[holes_created == np.min(holes_created)]

    def get_grids_after_available_actions(self, available_actions: np.ndarray) -> np.ndarray:
        """
        Returns the grids which result from applying each of the available actions.
        Each action is represented as 2 digits in base 10:
        - The first digit is the rotation (0-3).
        - The second digit is the column (0-9).
        """

        # Preallocate result grids and temporary grid, reused for simulating each action
        grids = np.zeros((len(available_actions), self.height, self.width), dtype=np.float32)
        tmp_grid = np.zeros_like(self.grid)
        tmp_tetromino = Tetromino(self.get_current_tetromino_type())

        for i, action_idx in enumerate(available_actions):

            np.copyto(tmp_grid, self.grid)
            rotation, col = divmod(action_idx, self.width)
            tmp_tetromino.spawn(x=col, rotation=rotation)

            # Hard drop
            while not self._intersects_stateless(tmp_grid, tmp_tetromino):
                tmp_tetromino.y += 1
            tmp_tetromino.y -= 1

            # Freeze
            x, y = tmp_tetromino.x, tmp_tetromino.y
            for cell_index in tmp_tetromino.image():
                row, col = y + (cell_index // 4), x + (cell_index % 4)
                tmp_grid[row][col] = 1

            # Break lines
            final_grid = self._break_lines_stateless(tmp_grid)

            # Copy resulting grid after applying the action
            np.copyto(grids[i], final_grid)

        return grids
