delate = []
gamma = 1
t=0
rewards=[1,2,3,4,5,6,7,8,9,10,[10]]
values = [1,2,3,4,5,6,7,8,9,10]
#* (1 - dones[t, :]) - values[t, :]
#delta = rewards[1, :] + gamma * values[1 + 1, :]
# print(rewards[1,:])
import numpy as np

a=np.array([8.,8.],dtype=np.float32)

b = np.pi*3/2
print (a)

[e,f,g] = [1,2,3]
print (e,f,g)
print g
print a

goal_list = [[-25.00, -0.00], [-24.80, -3.13], [-24.21, -6.22], [-23.24, -9.20], [-21.91, -12.04], [-20.23, -14.69],
             [-18.22, -17.11], [-15.94, -19.26], [-13.40, -21.11], [-10.64, -22.62], [-7.73, -23.78],
             [-4.68, -24.56], [-1.57, -24.95], [1.57, -24.95], [4.68, -24.56], [7.73, -23.78], [10.64, -22.62],
             [13.40, -21.11], [15.94, -19.26], [18.22, -17.11], [20.23, -14.69], [21.91, -12.04], [23.24, -9.20],
             [24.21, -6.22], [24.80, -3.13], [25.00, 0.00], [24.80, 3.13], [24.21, 6.22], [23.24, 9.20],
             [21.91, 12.04], [20.23, 14.69], [18.22, 17.11], [15.94, 19.26], [13.40, 21.11], [10.64, 22.62],
             [7.73, 23.78], [4.68, 24.56], [1.57, 24.95], [-1.57, 24.95], [-4.68, 24.56], [-7.73, 23.78],
             [-10.64, 22.62], [-13.40, 21.11], [-15.94, 19.26], [-18.22, 17.11], [-20.23, 14.69], [-21.91, 12.04],
             [-23.24, 9.20], [-24.21, 6.22], [-24.80, 3.13]
             ]
print len(goal_list)
init_pose_list = [[-7.00, 11.50, np.pi], [-7.00, 9.50, np.pi], [-18.00, 11.50, 0.00], [-18.00, 9.50, 0.00],
                  [-12.50, 17.00, np.pi * 3 / 2], [-12.50, 4.00, np.pi / 2], [-2.00, 16.00, -np.pi / 2],
                  [0.00, 16.00, -np.pi / 2],
                  [3.00, 16.00, -np.pi / 2], [5.00, 16.00, -np.pi / 2], [10.00, 4.00, np.pi / 2],
                  [12.00, 4.00, np.pi / 2],
                  [14.00, 4.00, np.pi / 2], [16.00, 4.00, np.pi / 2], [18.00, 4.00, np.pi / 2], [-2.5, -2.5, 0.00],
                  [-0.5, -2.5, 0.00], [3.5, -2.5, np.pi], [5.5, -2.5, np.pi], [-2.5, -18.5, np.pi / 2],
                  [-0.5, -18.5, np.pi / 2], [1.5, -18.5, np.pi / 2], [3.5, -18.5, np.pi / 2], [5.5, -18.5, np.pi / 2],
                  [-6.00, -10.00, np.pi], [-7.15, -6.47, np.pi * 6 / 5], [-10.15, -4.29, np.pi * 7 / 5],
                  [-13.85, -4.29, np.pi * 8 / 5],
                  [-16.85, -6.47, np.pi * 9 / 5], [-18.00, -10.00, np.pi * 2], [-16.85, -13.53, np.pi * 11 / 5],
                  [-13.85, -15.71, np.pi * 12 / 5],
                  [-10.15, -15.71, np.pi * 13 / 5], [-7.15, -13.53, np.pi * 14 / 5], [10.00, -17.00, np.pi / 2],
                  [12.00, -17.00, np.pi / 2],
                  [14.00, -17.00, np.pi / 2], [16.00, -17.00, np.pi / 2], [18.00, -17.00, np.pi / 2],
                  [10.00, -2.00, -np.pi / 2],
                  [12.00, -2.00, -np.pi / 2], [14.00, -2.00, -np.pi / 2], [16.00, -2.00, -np.pi / 2],
                  [18.00, -2.00, -np.pi / 2]]
print len(init_pose_list)
goal_list = [[-18.0, 11.5], [-18.0, 9.5], [-7.0, 11.5], [-7.0, 9.5], [-12.5, 4.0], [-12.5, 17.0],
             [-2.0, 3.0], [0.0, 3.0], [3.0, 3.0], [5.0, 3.0], [10.0, 10.0], [12.0, 10.0],
             [14.0, 10.0], [16.0, 10.0], [18.0, 10.0], [3.5, -2.5], [5.5, -2.5], [-2.5, -2.5],
             [-0.5, -2.5], [-2.5, -5.5], [-0.5, -5.5], [1.5, -5.5], [3.5, -5.5], [5.5, -5.5],
             [-18.0, -10.0], [-16.85, -13.53], [-13.85, -15.71], [-10.15, -15.71], [-7.15, -13.53], [-6.00, -10.00],
             [-7.15, -6.47], [-10.15, -4.29], [-13.85, -4.29], [-16.85, -6.47],
             ]
print len(goal_list)

init_pose_list = [[25.00, 0.00, np.pi], [24.80, 3.13, np.pi * 26 / 25], [24.21, 6.22, np.pi * 27 / 25],
                  [23.24, 9.20, np.pi * 28 / 25],
                  [21.91, 12.04, np.pi * 29 / 25], [20.23, 14.69, np.pi * 30 / 25], [18.22, 17.11, np.pi * 31 / 25],
                  [15.94, 19.26, np.pi * 32 / 25],
                  [13.40, 21.11, np.pi * 33 / 25], [10.64, 22.62, np.pi * 34 / 25],
                  [7.73, 23.78, np.pi * 35 / 25], [4.68, 24.56, np.pi * 36 / 25], [1.57, 24.95, np.pi * 37 / 25],
                  [-1.57, 24.95, np.pi * 38 / 25],
                  [-4.68, 24.56, np.pi * 39 / 25], [-7.73, 23.78, np.pi * 40 / 25],
                  [-10.64, 22.62, np.pi * 41 / 25], [-13.40, 21.11, np.pi * 42 / 25], [-15.94, 19.26, np.pi * 43 / 25],
                  [-18.22, 17.11, np.pi * 44 / 25],
                  [-20.23, 14.69, np.pi * 45 / 25], [-21.91, 12.04, np.pi * 46 / 25],
                  [-23.24, 9.20, np.pi * 47 / 25], [-24.21, 6.22, np.pi * 48 / 25], [-24.80, 3.13, np.pi * 49 / 25],
                  [-25.00, -0.00, np.pi * 50 / 25],
                  [-24.80, -3.13, np.pi * 51 / 25], [-24.21, -6.22, np.pi * 52 / 25], [-23.24, -9.20, np.pi * 53 / 25],
                  [-21.91, -12.04, np.pi * 54 / 25], [-20.23, -14.69, np.pi * 55 / 25],
                  [-18.22, -17.11, np.pi * 56 / 25], [-15.94, -19.26, np.pi * 57 / 25],
                  [-13.40, -21.11, np.pi * 58 / 25],
                  [-10.64, -22.62, np.pi * 59 / 25], [-7.73, -23.78, np.pi * 60 / 25],
                  [-4.68, -24.56, np.pi * 61 / 25], [-1.57, -24.95, np.pi * 62 / 25], [1.57, -24.95, np.pi * 63 / 25],
                  [4.68, -24.56, np.pi * 64 / 25], [7.73, -23.78, np.pi * 65 / 25], [10.64, -22.62, np.pi * 66 / 25],
                  [13.40, -21.11, np.pi * 67 / 25], [15.94, -19.26, np.pi * 68 / 25], [18.22, -17.11, np.pi * 69 / 25],
                  [20.23, -14.69, np.pi * 70 / 25], [21.91, -12.04, np.pi * 71 / 25], [23.24, -9.20, np.pi * 72 / 25],
                  [24.21, -6.22, np.pi * 73 / 25], [24.80, -3.13, np.pi * 74 / 25]
                  ]
print len(init_pose_list)
