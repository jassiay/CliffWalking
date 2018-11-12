import numpy as np
import random
import matplotlib.pyplot as plt

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 1
LEN = 12
WID = 4

action_dest = []
for i in range(0, 12):
    action_dest.append([])
    for j in range(0, 4):
        destination = dict()
        destination[0] = [i, min(j+1,3)]
        destination[1] = [min(i+1,11), j]
        if 0 < i < 11 and j == 1:
            destination[2] = [0,0]
        else:
            destination[2] = [i, max(j - 1, 0)]
        destination[3] = [max(i-1,0), j]
        action_dest[-1].append(destination)
action_dest[0][0][1] = [0,0]

action_reward = np.zeros((LEN, WID, 4))
action_reward[:, :, :] = -1.0
action_reward[1:11, 1, 2] = -100.0
action_reward[0, 0, 1] = -100.0

def take_step(x,y,a):
    goal = 0
    if x == LEN - 1 and y == 0:
        goal = 1
    if a == 0:
        y += 1
    if a == 1:
        x += 1
    if a == 2:
        y -= 1
    if a == 3:
        x -= 1
        
    x = max(0,x)
    x = min(LEN-1, x)
    y = max(0,y)
    y = min(WID-1, y)

    if goal == 1:
        return x,y,-1
    if x>0 and x<LEN-1 and y==0:
        return 0,0,-100
    return x,y,-1

def epsGreedyPolicy(x,y,q,eps):
    t = random.randint(0,3)
    if random.random() < eps:
        a = t
    else:
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
    return a

def findMaxQ(x,y,q):
    q_max = q[x][y][0]
    a_max = 0
    for i in range(4):
        if q[x][y][i] >= q_max:
            q_max = q[x][y][i]
            a_max = i
    a = a_max
    return a

def sarsa(q):
    runs = 30
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            a = epsGreedyPolicy(x,y,q,EPSILON)
            while True:
                [x_next,y_next] = action_dest[x][y][a]
                reward = action_reward[x][y][a]
                reward_sum += reward
                a_next = epsGreedyPolicy(x_next,y_next,q,EPSILON)
                q[x][y][a] += ALPHA*(reward + GAMMA*q[x_next][y_next][a_next]-q[x][y][a])
                if x == LEN - 1 and y==0:
                    break
                x = x_next
                y = y_next
                a = a_next
            rewards[i] += reward_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards

def qLearning(q):
    runs = 30
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            reward_sum = 0
            x = 0
            y = 0
            while True:
                a = epsGreedyPolicy(x,y,q,EPSILON)             
                x_next, y_next,reward = take_step(x,y,a)
                a_next = findMaxQ(x_next,y_next,q)
                reward_sum += reward
                q[x][y][a] += ALPHA*(reward + GAMMA*q[x_next][y_next][a_next]-q[x][y][a])
                if x == LEN - 1 and y==0:
                    break
                x = x_next
                y = y_next
            rewards[i] += reward_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards

def showOptimalPolicy(q):
    for j in range(WID-1,-1,-1):
        for i in range(LEN):
            a = findMaxQ(i,j,q)
            if a == 0:
                print(" U ",end = "")
            if a == 1:
                print(" R ",end = "")
            if a == 2:
                print(" D ",end = "")
            if a == 3:
                print(" L ",end = "")
        print("")

def showOptimalPath(q):
    x = 0
    y = 0
    path = np.zeros([LEN,WID]) - 1
    end = 0
    exist = np.zeros([LEN,WID])
    while (x != LEN-1 or y != 0) and end == 0:
        a = findMaxQ(x,y,q)
        path[x][y] = a
        if exist[x][y] == 1:
            end = 1
        exist[x][y] = 1
        x,y,r = take_step(x,y,a)
    for j in range(WID-1,-1,-1):
        for i in range(LEN):
            if i == 0 and j == 0:
                print(" S ",end = "")
                continue
            if i == LEN-1 and j == 0:
                print(" G ",end = "")
                continue
            a = path[i,j]
            if a == -1:
                print(" - ",end = "")
            elif a == 0:
                print(" U ",end = "")
            elif a == 1:
                print(" R ",end = "")
            elif a == 2:
                print(" D ",end = "")
            elif a == 3:
                print(" L ",end = "")
        print("")

s_grid = np.zeros([12,4,4])
sarsa_rewards = sarsa(s_grid)

q_grid = np.zeros([12,4,4])
q_learning_rewards = qLearning(q_grid)

plt.plot(range(len(sarsa_rewards)),sarsa_rewards,label="sarsa")
plt.plot(range(len(sarsa_rewards)),q_learning_rewards,label="Qlearning")
plt.ylim(-100,0)
plt.legend(loc="lower right")
plt.show()

print("Sarsa Optimal Policy")
showOptimalPolicy(s_grid)
print("Q-learning Optimal Policy")
showOptimalPolicy(q_grid)

print("Sarsa Optimal Path")
showOptimalPath(s_grid)
print("Q-learning Optimal Path")
showOptimalPath(q_grid)