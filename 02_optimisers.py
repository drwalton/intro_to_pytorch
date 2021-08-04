import torch

# Generally when actually optimising something, it's easiest to use one of pytorch's built-in optimisers

tensor_to_optimise = torch.ones((1), requires_grad=True)

# There are a variety of optimisers, but all are initialised with an array of tensors you'd like optimised
# Make sure all of these have requires_grad set to True
# Most optimisers have a learning rate or other parameters, for SGD (stochastic gradient descent)
# here you can set the learning rate as follows:
optimiser = torch.optim.SGD([tensor_to_optimise], lr=0.1)

# Now we'll minimise the distance between tensor_to_optimise and this target tensor
target_tensor = torch.zeros((1))

# Here's a basic optimisation loop
loss_func = torch.nn.MSELoss()

for iter in range(10):
    # First zero gradients from the previous iteration
    optimiser.zero_grad()

    # Do any operations in your forward pass, and find a loss
    loss = loss_func(tensor_to_optimise, target_tensor)
    print("Iteration %d loss %f" % (iter, loss.item())) # Here item() converts the tensor to a float

    # Find the gradients with loss.backward()
    loss.backward()

    # Advance the optimiser one step using these gradients
    optimiser.step()

# After this our tensor should be closer to the zero target tensor
print("Final tensor", tensor_to_optimise.item())
