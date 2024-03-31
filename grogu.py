import torch

print(torch.cuda.is_available())
pytorch_device = torch.device('cuda:0') # uses the first GPU available right now
print(pytorch_device)

def main(args): # args should contain informatin about the path of the configuration fie
    pytorch_device = torch.device('cuda:0') # uses the first GPU available right now. returns 'cuda:0'

    config_path = args.config_path # returns the path of the config file
    
    configs = load_config_data(config_path) # loads the 

