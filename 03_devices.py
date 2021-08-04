import torch

# Another appeal of torch is that it makes it very easy to perform many tasks on the GPU.

tensor = torch.zeros((1))

# You can move any tensor to the GPU with .cuda()
cuda_tensor = tensor.cuda()

# You can move anything back to the CPU with .cpu()
cpu_tensor = cuda_tensor.cpu()

# If the tensor was already on the CPU, it won't throw an error:
cpu_tensor = tensor.cpu()

# You can also make a device object, and use the .to() function:
device = torch.device("cuda:0")

cuda_tensor = tensor.to(device)

# There are advantages to this: first you can change the index to use other GPUs if you have them.
# Also you can create tensors on the right device initially, without having to transfer them:

cuda_tensor = torch.zeros((1), device=device)

# It's also handy for making running on GPU optional in your program, by adding this to the top:

use_cuda = True

if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

my_tensor = torch.rand((1), device=device)

# You can't perform operations between CPU and GPU tensors without getting an error:

try:
    result = cuda_tensor * cpu_tensor    
except:
    print("The multiplication failed.")

# So it's important to make sure your tensors are all on the right device. 

# Generally only tensors or other objects that store data need to be on the correct device though.
# Things like loss functions don't need to be moved to the GPU, even if you're working on the GPU

loss_func = torch.nn.MSELoss()
print(loss_func(cuda_tensor, cuda_tensor))

# Remember that if you want to e.g. display an image in matplotlib, you need to move it back to the 
# CPU.
