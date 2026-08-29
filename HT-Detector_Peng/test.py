import torch


print(torch.backends.cudnn.version())

cuda_available = torch.cuda.is_available()

print('cuda_available=', cuda_available)
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.current_device())

tensor = torch.ones((2,), dtype=torch.int8)
print(tensor)
mybox = torch.tensor(data=[1, 2, 3, 4], device='cuda:0')
print(mybox)

#--