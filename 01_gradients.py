import torch

# The primary reason to use torch is for its automatic gradient calculation facility (autograd).
# This works fairly simply - you just make sure that the variables you want to optimise have
# the requires_grad property set to True

tensor_to_optimise = torch.ones((1), requires_grad=True)

# You can also set requires_grad to true after creating the tensor

tensor_to_optimise_2 = torch.ones((1))
tensor_to_optimise_2.requires_grad = True

# By default tensors have requires_grad set to False

new_tensor = torch.ones((1))
print(new_tensor.requires_grad)

# After this, you can perform some operations on the thing you want to optimise, and find a loss

target_tensor = torch.zeros((1))

modified_tensor = tensor_to_optimise # I'm leaving the tensor alone here, but you could perform any operation with it.
loss_function = torch.nn.L1Loss() # Here we're using L1 loss, i.e. sum of absolute distances.

loss = loss_function(target_tensor, modified_tensor)

# Finding the gradient is as easy as
loss.backward()

# You can find the value of the gradient with
print(tensor_to_optimise.grad)
