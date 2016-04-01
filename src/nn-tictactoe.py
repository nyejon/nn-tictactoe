import numpy
import theano
import theano.tensor as T
rng = numpy.random


#              0 | 1 | 2
#             ---+---+---
#              3 | 4 | 5
#             ---+---+---
#              6 | 7 | 8

# board = [0, ..., 8]
# board = [0,0,0,0,0,1,1,2,2] with 1 = O and 2 = X

# victory lines:
horizontal = [range(i, i + 3) for i in xrange(0, 9, 3)]
vertical = [range(i, i + 9, 3) for i in xrange(0, 3)]
diagonals = [[0, 4, 8], [2, 4, 6]]
victory = horizontal + vertical + diagonals

# board = [1,1,1,0,0,0,2,2,0]
board = [0, 0, 0, 0, 0, 0, 0, 0, 0]


def check_victory():
    # check if any of the victory lines have been filled by a single player
    if any(all(board[j] == 1 for j in i) for i in victory):
        return (1)  # player 1 wins
    elif any(all(board[j] == 2 for j in i) for i in victory):
        return (2)  # player 2 wins
    elif all(board[i] != 0 for i in xrange(9)):
        return (-1)  # draw
    return (0)  # no winner yet


def play_move(board_probability, player):
    # get the moves in order of highest rating / probability
    moves = sorted(enumerate(board_probability), key=lambda x: x[1], reverse=True)
    # try and play each of the moves from best to worst
    for move in moves:
        if board[move[0]] == 0:
            board[move[0]] = player
            return


def draw_board():
    def convert_output(state):
        if state == 0:
            return (" ")
        elif state == 1:
            return ("O")
        else:
            return("X")

    print ("%s|%s|%s" % (convert_output(board[0]), convert_output(board[1]), convert_output(board[2])))
    print ("------")
    print ("%s|%s|%s" % (convert_output(board[3]), convert_output(board[4]), convert_output(board[5])))
    print ("------")
    print ("%s|%s|%s" % (convert_output(board[6]), convert_output(board[7]), convert_output(board[8])))

player = 1
while check_victory() == 0:
    draw_board()
    board_probability = [rng.ranf() for i in xrange(9)]
    print("")
    play_move(board_probability, player)
    player += 1
    if player > 2:
        player = 1

draw_board()

print ("=" * 80)

victor = check_victory()
if victor in (1, 2):
    print ("player %s wins" % victor)
else:
    print (victor, "it was a draw")


