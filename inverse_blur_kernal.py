import numpy as np

# Define the coefficients of the system of equations
kernal = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Size of the padding (n)
padding_size = 10_000

# New dimensions of the kernel with padding
new_rows = kernal.shape[0] + padding_size
new_cols = kernal.shape[1] + padding_size

# Create a new padded kernel with zeros
padded_kernel = np.zeros((new_rows, new_cols))
padded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel

# Define the right-hand side of the equations
rhs = np.eye(3 + padding_size)

# Solve the system of equations
inverse_matrix = np.linalg.solve(padded_kernel, rhs)

print("Inverse Matrix:")
print(inverse_matrix)