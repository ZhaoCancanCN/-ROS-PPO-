import torch
import numpy as np
# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data)

# a = torch.arange(1,17)#1row   17 cloum
# print(a)
# print(a.size(0))
# print(a.shape)
# print (
#     "\n",np_data, "\n ",torch_data
# )
# pb = a.view(a.size(0),-1)
# print(b)rint("I'm Bob.\nWhat's your name?")

# b = a.view(a.size(0),-1)
# print(b)
#input.view(input.shape[0], 1,  -1)


a= torch.ones(3,2)
print a
b= 2*torch.ones(3,5)
print b
# print (torch.cat((a,b),0))
print (torch.cat((a,b),1))
print (torch.cat((a,b),-1))