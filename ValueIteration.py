import numpy as np
import time
import math


def read_input_file(file_name):
    with open(file_name, 'r') as input_file:
        lines = input_file.readlines()
        size = int(lines[0].rstrip())  # size of grid
        no_of_cars = int(lines[1].rstrip())  # number of cars
        cars = [[None for col in range(2)] for row in range(no_of_cars)]
        ends = [[None for col in range(2)] for row in range(no_of_cars)]
        no_of_obstacles = int(lines[2].rstrip())  # number of obstacles
        rewards = [[-1.0 for col in range(size)] for row in range(size)]
        total_no_of_lines = no_of_obstacles + 3
        for i in range(3, total_no_of_lines):
            x, y = (lines[i].rstrip().split(","))
            rewards[int(y)][int(x)] = -101.0
        for i in range(total_no_of_lines, total_no_of_lines + no_of_cars):
            x, y = (lines[i].rstrip().split(","))
            cars[i - total_no_of_lines][0] = int(y)
            cars[i - total_no_of_lines][1] = int(x)
        total_no_of_lines = total_no_of_lines + no_of_cars
        j = 0
        for i in range(total_no_of_lines, total_no_of_lines + no_of_cars):
            x, y = (lines[i].rstrip().split(","))
            ends[i - total_no_of_lines][0] = int(y)
            ends[i - total_no_of_lines][1] = int(x)
            j += 1
        return size, rewards, cars, ends


def write_output_file(output):
    with open('output.txt', 'w') as output_file:
        output_string = '\n'.join(output)
        output_file.write(output_string)


def move_from(i, j, move, size):
    if move == 0 and i - 1 >= 0:
        return i - 1, j
    elif move == 1 and i + 1 < size:
        return i + 1, j
    elif move == 2 and j + 1 < size:
        return i, j + 1
    elif move == 3 and j - 1 >= 0:
        return i, j - 1
    else:
        return i, j


def generate_random_variables():
    list_of_swerve = []
    for j in range(10):
        np.random.seed(j)
        swerve = np.random.random_sample(1000000)
        list_of_swerve.append(swerve)
    return list_of_swerve


def simulation(cars, ends, cache, rewards, hundred_point_car, size):
    back_move, left_move, right_move = move_dictionaries()
    swerve = generate_random_variables()
    output = []
    for i in range(len(cars)):
        if i in hundred_point_car:
            output.append(str(100))
            continue
        key = str(ends[i][0]) + "_" + str(ends[i][1])
        end_value = rewards[ends[i][0]][ends[i][1]]
        rewards[ends[i][0]][ends[i][1]] = 99.0
        p = cache[key]
        solution = 0
        for j in range(10):
            pos = [cars[i][0], cars[i][1]]
            k = 0
            final_reward = 0
            while not (pos[0] == ends[i][0] and pos[1] == ends[i][1]):
                move = p[pos[0]][pos[1]]
                if swerve[j][k] > 0.7:
                    if swerve[j][k] > 0.8:
                        if swerve[j][k] > 0.9:
                            move = back_move[move]
                        else:
                            move = left_move[move]
                    else:
                        move = right_move[move]
                pos[0], pos[1] = move_from(pos[0], pos[1], move, size)
                final_reward += rewards[pos[0]][pos[1]]
                k += 1
            solution += final_reward
        output.append(str(int(math.floor(solution / 10.0))))
        rewards[ends[i][0]][ends[i][1]] = end_value
    return output


def move_dictionaries():
    left_move = dict()
    left_move[0] = 2
    left_move[1] = 3
    left_move[2] = 1
    left_move[3] = 0
    back_move = dict()
    back_move[0] = 1
    back_move[1] = 0
    back_move[2] = 3
    back_move[3] = 2
    right_move = dict()
    right_move[0] = 3
    right_move[1] = 2
    right_move[2] = 0
    right_move[3] = 1
    return back_move, left_move, right_move


def calculate_action_probablities(i, j, size, utitlity1):
    pos_prob = 0.7
    neg_prob = 0.1
    if i - 1 >= 0:
        a = utitlity1[i - 1][j]
    else:
        a = utitlity1[i][j]
    if i + 1 < size:
        b = utitlity1[i + 1][j]
    else:
        b = utitlity1[i][j]
    if j + 1 < size:
        c = utitlity1[i][j + 1]
    else:
        c = utitlity1[i][j]
    if j - 1 >= 0:
        d = utitlity1[i][j - 1]
    else:
        d = utitlity1[i][j]
    m = pos_prob * a + neg_prob * (b + c + d)
    n = pos_prob * b + neg_prob * (a + c + d)
    o = pos_prob * c + neg_prob * (a + b + d)
    p = pos_prob * d + neg_prob * (a + b + c)
    return m, n, o, p


def main(f):
    cache = dict()
    hundred_point_car = dict()
    size, reward, cars, ends = read_input_file(f)
    gamma = 0.9
    epsilon = 0.1
    value_check = epsilon*((1.0-gamma)/gamma)
    for c in range(len(cars)):
        end_x = ends[c][0]
        end_y = ends[c][1]
        start_x = cars[c][0]
        start_y = cars[c][1]
        if end_x == start_x and end_y == start_y:
            hundred_point_car[c] = 100
            continue
        key = str(end_x) + "_" + str(end_y)
        if key in cache:
            continue
        end_value = reward[end_x][end_y]
        reward[end_x][end_y] = 99.0
        policy = [[-1.0 for col in range(size)] for row in range(size)]
        utitlity = [[reward[i][j] for j in range(size)] for i in range(size)]
        while True:
            diff = 0.0
            utitlity1 = [[utitlity[i][j] for j in range(size)] for i in range(size)]
            for i in range(0, size):
                for j in range(0, size):
                    if i == end_x and j == end_y:
                        continue
                    action_value = calculate_action_probablities(i, j, size, utitlity1)
                    max_value = max(action_value)
                    utitlity[i][j] = reward[i][j] + gamma * max_value
                    x = utitlity[i][j] - utitlity1[i][j]
                    x = abs(x)
                    if x > diff:
                        diff = x
            if diff < value_check:
                for i in range(size):
                    for j in range(size):
                        action_value = calculate_action_probablities(i, j, size, utitlity1)
                        max_value = max(action_value)
                        policy[i][j] = action_value.index(max_value)
                cache[key] = policy
                break
        reward[end_x][end_y] = end_value
    output = simulation(cars, ends, cache, reward, hundred_point_car, size)
    print output
    write_output_file(output)


if __name__ == "__main__":
    t = time.time()
    main("input.txt")
    print (time.time()) - t