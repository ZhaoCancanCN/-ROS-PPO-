import numpy as np
# list = [[0,1,2,3],[0,1,2,3]]
# print(list)
# list2 = np.asarray(list)
# print(list2)
# print(list2.shape)
num_step = 8
# reward = [1,2,3,4,5,6,7,8,9]
# value =  [1,1,1,1,1,1,1,1,1]
# for t in range(num_step - 1, -1, -1):
#     delta =reward[t, :]-value[t, :]
#     print delta
reward = [[1,2],[3,4]]
value = [[2,2],[2,2]]  #
reward = np.asarray(reward)
value = np.asarray(value)
# print reward.mean()
# print reward
# for t in range(2,-1,-1):
#     print t
#     print(reward[t:]-value[t:])
# gae = np.zeros((4,))
# print gae
# print reward
# adv = (reward.reshape(4,1))
# print adv
# adv = (value.mean())
# print adv
import torch
a=torch.Tensor([[[1,2,3],[4,5,6]]])
b=torch.Tensor([1,2,3,4,5,6])

print(a.view(-1,6))
print(b.view(1,6))

a=torch.Tensor([[[1,2,3],[4,5,6]]])
print(a.view(1,-1))