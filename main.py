import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from check_submission import check_submission
from game_mechanics import (
    get_legal_moves,
    make_move,
    choose_move_randomly,
    OthelloEnv,
    play_othello_game,
    load_network,
    save_network,
)

TEAM_NAME = "Connected_Pawns"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size = 2)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size = 2)
        self.fc1 = torch.nn.Linear(16*32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.do1 = torch.nn.Dropout(0.1)
        self.do2 = torch.nn.Dropout(0.1)
        self.do3 = torch.nn.Dropout(0.1)
        self.do4 = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.do1(x)
        x = self.relu2(self.conv2(x))
        x = self.do2(x)
        x = x.view(1, 16*32)
        x = nn.functional.relu(self.fc1(x))
        x = self.do3(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.do4(x)
        x = self.fc3(x)
        return(x)

epsilon = 0
alpha = 0.1
gamma = 0.95
delta = 0

def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pytorch network to be used by choose_move. You can architect
        this however you like but your choose_move function must be able
        to use it.
    """
    # net = SimpleCNN()
    net = load_network(TEAM_NAME)
    env = OthelloEnv()
    # print("Weights before training (randomly initialised):\n", net.state_dict())

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    cum_loss = 0
    
    for i in range(1000000):
        total_loss = 0
        num_turns = 0
        state, reward, done, info = env.reset()
        while not done:
            ep = random.choices([0, 1], [epsilon * 10, (1 - epsilon) * 10])
            if ep == [0]:    # exploration
                action = choose_move_randomly(state)
            else:            # exploitation
                action = choose_move(state, net)
            new_state, reward, done, _ = env.step(action)
            old_state_value = net.forward(torch.Tensor(np.expand_dims(state, 0)))
            old_state_new_value = alpha * (reward + gamma*net.forward(torch.Tensor(np.expand_dims(new_state, 0)))-old_state_value )
            state = new_state
        
            # Calculate the loss of the network at this point in time over all the data
            loss = loss_fn(old_state_value, old_state_new_value)
            total_loss += loss
            num_turns += 1

            # Reset the gradient in the optimizer
            optimizer.zero_grad()

            # Use the loss function to figure out the direction to move the parameters
            loss.backward()
            
            # Update parameters to reduce loss function (and improve the network!)
            optimizer.step()
        
        new_state_value = net.forward(torch.Tensor(np.expand_dims(state, 0)))
        
        # Calculate the loss of the network at this point in time over all the data
        loss = loss_fn(new_state_value, torch.Tensor([[reward]]))
        total_loss += loss
        num_turns += 1

        # Reset the gradient in the optimizer
        optimizer.zero_grad()

        # Use the loss function to figure out the direction to move the parameters
        loss.backward()
        
        # Update parameters to reduce loss function (and improve the network!)
        optimizer.step()

        if (i%100==0):
            print (cum_loss)
            cum_loss = 0
            save_network(net, TEAM_NAME)
        else:
            cum_loss += total_loss/num_turns
         
    return net

def choose_move(state: np.ndarray,
                neural_network: nn.Module,
                verbose: bool = False) -> Optional[Tuple[int, int]]:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state: State of the board as a np array. Your pieces are
                1's, the opponent's are -1's and empty are 0's.
        network: The pytorch network output by train().
        verbose: Whether to print debugging information to console.

    Returns:
        position (Tuple | None): The position (row, col) you want to place
                        your counter. row and col should be an integer 0 -> 7)
                        where (0,0) is the top left corner and
                        (7,7) is the bottom right corner.
                        You should return None if no legal move is available
    """
    legal_moves = get_legal_moves(state)
    if verbose:
        print(legal_moves)
    if not legal_moves:
        return None

    max = -math.inf
    max_move = 0,0

    for move in legal_moves:
        state_copy = state.copy()
        make_move(state_copy, move)

        # print (torch.Tensor(state_copy))
        state_copy = np.expand_dims(state_copy, 0)

        value = neural_network.forward(torch.Tensor(state_copy))
        # print (value.shape)
        if value[0][0] > max:
            max_move = move
            max = value
    # print (move)
    # print (value)
    return max_move

if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    # my_network = train()
    # save_network(my_network, TEAM_NAME)
    my_network = load_network(TEAM_NAME)
    print (my_network)

    # Code below plays a single game of Connect 4 against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(
            state: np.ndarray) -> Optional[Tuple[int, int]]:
        """The arguments in play_connect_4_game() require functions that only take the state as
        input.

        This converts choose_move() to that format.
        """
        return choose_move(state, my_network, False)

    play_othello_game(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=10,
        render=True,
        verbose=True,
    )

    # Uncomment line below to check your submission works
    check_submission()
