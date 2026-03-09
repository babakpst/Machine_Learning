import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)


print(torch.version.cuda)




x = torch.rand(3,3).cuda()
print(x)
