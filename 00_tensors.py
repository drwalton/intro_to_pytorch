import torch

# Torch stores most data in tensors, which behave mostly similarly to numpy arrays.
tensor_0 = torch.Tensor([1])
tensor_1 = torch.Tensor([[0, 1]])

# One slight difference is that instead of .shape, torch uses .size(), a function
print(tensor_0.size()) # Returns the size of all the dimensions in a tuple
print(tensor_1.size(0)) # Prints the size of the 0th dimension of the tensor

# Tensors can also be initialised with constant values, random noise etc.
tensor_zeros = torch.zeros((512,512))
tensor_ones = torch.ones((1,1,1,32))
tensor_rand = torch.rand((42))

# Tensors representing images are generally formatted in NCHW format.
# The first dimension N is the batch dimension. It's common to process multiple images in a batch
# as this improves efficiency and can make training more stable.
# The second dimension C contains the channels (e.g. red, green and blue for an RGB image)
# The final two dimensions H and W contain the height and width.

# For example, to make a tensor to store a batch of 5 1920x1080 RGB images, you could declare:
image_batch = torch.zeros((5,3,1080,1920))

# It isn't required you store all images in NCHW format, but many built-in torch functions will
# expect input in this form.
