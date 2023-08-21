from ms.game import Game

N = 10

g = Game(N=N)

marker = (0, 0)
flags = set()

lost = False
while not lost:
    g.print_board(marker=marker, flags=flags)

    move = input("Enter your move(l,r,d,u to move / c to click / f,r to flag): ")

    for move_k in move.strip():
        if move_k == 'u':
            marker = (max(marker[0]-1, 0), marker[1])
        elif move_k == 'd':
            marker = (min(marker[0]+1, N-1), marker[1])
        elif move_k == 'l':
            marker = (marker[0], max(marker[1]-1, 0))
        elif move_k == 'r':
            marker = (marker[0], min(marker[1]+1, N-1))

        elif move_k == 'c':
            lost = g.click(marker[0], marker[1])

        elif move_k == 'f':
            flags.add(marker)
        elif move_k == 'r':
            flags.remove(marker)

