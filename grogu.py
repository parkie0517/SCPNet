import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

var = None
if not isinstance(var, int) or var <= 0:
    print(var)