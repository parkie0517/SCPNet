import torch

print(torch.cuda.is_available())

epoch = 12
print('final_model_'+str(epoch)+'.pth')

from tensorboardX import SummaryWriter
writer = SummaryWriter('./logs/') # Write training results in './logs/' directory

writer.add_scalar("Loss/train", 3.121, epoch) # tag, value, step
epoch+=1
writer.add_scalar("Loss/train", 3.121, epoch) # tag, value, step
epoch+=1
writer.add_scalar("Loss/train", 3.121, epoch) # tag, value, step
epoch+=1
writer.add_scalar("Loss/train", 3.121, epoch) # tag, value, step