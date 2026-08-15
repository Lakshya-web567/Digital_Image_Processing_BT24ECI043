import random

ticket = [[0]*9 for _ in range(3)]
valid = False

while valid == False:

    position = [[0]*9 for _ in range(3)]

    for i in range(3):
        p = random.sample(range(9), 5)
        for col in p:
            position[i][col] = 1

    count = [sum(position[i][j] for i in range(3)) for j in range(9)]

    if all(c >= 1 for c in count) and all(c <= 3 for c in count):
        valid = True

for j in range(9):

    if j == 0:
        rng = list(range(1, 10))
    elif j == 1:
        rng = list(range(10, 20))
    elif j == 2:
        rng = list(range(20, 30))
    elif j == 3:
        rng = list(range(30, 40))
    elif j == 4:
        rng = list(range(40, 50))
    elif j == 5:
        rng = list(range(50, 60))
    elif j == 6:
        rng = list(range(60, 70))
    elif j == 7:
        rng = list(range(70, 80))
    else:
        rng = list(range(80, 91))

    numbers = random.sample(rng, sum(position[i][j] for i in range(3)))
    numbers.sort()

    k = 0
    for i in range(3):
        if position[i][j] == 1:
            ticket[i][j] = numbers[k]
            k = k + 1

for i in range(3):
    line = ""
    for j in range(9):
        if ticket[i][j] == 0:
            line = line + "  . "
        else:
            line = line + str(ticket[i][j]).rjust(3) + " "
    print(line)