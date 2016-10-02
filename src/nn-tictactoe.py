import numpy as np
import theano
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from deap import base
from deap import tools
rng = np.random


#              0 | 1 | 2
#             ---+---+---
#              3 | 4 | 5
#             ---+---+---
#              6 | 7 | 8

# board = [0, ..., 8]
# board = [0,0,0,0,0,1,1,2,2] with 1 = O and 2 = X

# victory lines:
horizontal = [range(i, i + 3) for i in range(0, 9, 3)]
vertical = [range(i, i + 9, 3) for i in range(0, 3)]
diagonals = [[0, 4, 8], [2, 4, 6]]
victory = horizontal + vertical + diagonals

weight_shapes = []

# define our two players, X and O

# Dense(X) is a fully-connected layer with X hidden units.
# in the first layer, you must specify the expected input data shape:
# here, a 9-dimensional vector.
# player_1 = Sequential()
# player_1.add(Dense(20, input_dim=9, init='uniform', activation='tanh'))
# player_1.add(Dense(20, init='uniform', activation='tanh'))
# player_1.add(Dense(9, init='uniform', activation='tanh'))
# player_1.compile(optimizer='sgd', loss='mse')
# [i.shape for i in player_1.get_weights()]


def create_player():
    player = Sequential()
    player.add(Dense(20, input_dim=9, init='uniform', activation='tanh'))
    player.add(Dense(20, init='uniform', activation='tanh'))
    player.add(Dense(9, init='uniform', activation='tanh'))
    player.compile(optimizer='sgd', loss='mse')
    return player


def check_victory(board):
    # check if any of the victory lines have been filled by a single player
    if any(all(board[j] == 1 for j in i) for i in victory):
        return 1  # player 1 wins
    elif any(all(board[j] == 2 for j in i) for i in victory):
        return 2  # player 2 wins
    elif all(board[i] != 0 for i in range(9)):
        return -1  # draw
    return 0  # no winner yet


def play_move(board_probability, player, board):
    # get the moves in order of highest rating / probability
    moves = sorted(enumerate(board_probability), key=lambda x: x[1], reverse=True)
    # try and play each of the moves from best to worst
    for move in moves:
        if board[move[0]] == 0:
            board[move[0]] = player
            return


def draw_board(board):
    def convert_output(state):
        if state == 1:
            return "O"
        elif state == 2:
            return "X"
        else:
            return " "

    print("%s|%s|%s" % (convert_output(board[0]), convert_output(board[1]), convert_output(board[2])))
    print("------")
    print("%s|%s|%s" % (convert_output(board[3]), convert_output(board[4]), convert_output(board[5])))
    print("------")
    print("%s|%s|%s" % (convert_output(board[6]), convert_output(board[7]), convert_output(board[8])))
    print("\n")


def convert_to_vector(model):
    weights = model.get_weights()
    unrolled = []
    global weight_shapes
    weight_shapes = []
    for weight in weights:
        unrolled.append((weight.T).ravel())
        weight_shapes.append(weight.shape)
    vector = np.concatenate(unrolled)
    return vector


"""
vector = convert_to_vector(player_1)
weights = convert_to_matrix(vector)
"""


def convert_to_matrix(vector):
    sizes = []
    sizeTot = 0
    for weight in weight_shapes:
        if len(weight) > 1:
            size = weight[0]*weight[1]
        else:
            size = weight[0]
        sizeTot += size
        sizes.append(sizeTot)

    rolled = np.split(vector, sizes)
    reshaped = []
    for roll, shape in zip(rolled, weight_shapes):
        new = np.reshape(roll, shape)
        reshaped.append(new)
    return reshaped


def AI(player, board):
    return player.predict(np.array([board]))[0]


def play_game(player_1, player_2):
    player = 1
    victor = 0
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # create a fresh board for the game
    while victor == 0:
        # draw_board(board)
        player_model = player_1 if player == 1 else player_2
        board_probability = AI(player_model, board)
        play_move(board_probability, player, board)
        # check for victory
        victor = check_victory(board)
        # rotate the player:
        player += 1
        if player > 2:
            player = 1
    draw_board(board)
    return victor


def evaluate(player_roster):
    # every player will play every other player and the scores will be tallied
    scores = [0 for i in range(len(player_roster))]
    for i, player_1 in enumerate(player_roster):
        for j, player_2 in enumerate(player_roster):
            # don't play yourself bro
            if i == j:
                continue
            print("*" * 80)
            print("NOW FIGHTING: Players {} and {}".format(i, j))
            print("*" * 80)
            result = best_of_10(player_1, player_2)
            print("RESULTING SCORES: {} and {}".format(result[0], result[1]))
            print("\n\n")
            # the scores are a vector, add each player's score to their total
            scores[i] += result[0]
            scores[j] += result[1]
    print("\n\n\nAND THE FINAL SCORES ARE:")
    print(scores)


def best_of_10(player_1, player_2):
    score = [0, 0]  # [player 1, player 2]
    # the players take turns going first (5 each)
    # unfortunately, at the moment the bots never play differently so there's no point in a BO10
    # --> need to fix this later by tweaking the AI() function
    # for i in xrange(5):
    victor = play_game(player_1, player_2)
    if victor > 0:
        score[victor - 1] += 1
    # for i in xrange(5):
    victor = play_game(player_2, player_1)
    if victor > 0:
        score[victor - 1] += 1
    return score

player_roster = [create_player() for i in range(10)]  # our players
evaluate(player_roster)

print("=" * 80)

# victor = check_victory()
# if victor in (1, 2):
#    print ("player %s wins" % victor)
# else:
#    print (victor, "it was a draw")
