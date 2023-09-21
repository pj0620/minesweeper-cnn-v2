# from ms.game import Game
from game import Game

N = 10

g = Game(N=N, seed=4)

marker = (0, 0)
flags = set()

lost = False
while not lost:
    g.print_board(marker=marker, flags=flags)

    move = input("Enter your move(l,r,d,u to move / c to click / f,r to flag): ")

    game_over = False
    win_lose = None

    for move_k in move.strip():
        if move_k == 'u':
            marker = (max(marker[0] - 1, 0), marker[1])
        elif move_k == 'd':
            marker = (min(marker[0] + 1, N - 1), marker[1])
        elif move_k == 'l':
            marker = (marker[0], max(marker[1] - 1, 0))
        elif move_k == 'r':
            marker = (marker[0], min(marker[1] + 1, N - 1))

        elif move_k == 'c':
            game_over, win_lose = g.click(marker[0], marker[1])

        elif move_k == 'f':
            flags.add(marker)
        elif move_k == 'r':
            flags.remove(marker)

    if game_over:
        print("You won :)" if win_lose == "win" else "You lose :(")
        break
