import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_stable = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_stable)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
             # 1. Move softmax dim to last position (N, ..., C)
            self.A = np.moveaxis(self.A, self.dim, -1)
            dLdA = np.moveaxis(dLdA, self.dim, -1)

            # 2. Flatten to (batch_size, C)
            A_2d = self.A.reshape(-1, C)
            dLdA_2d = dLdA.reshape(-1, C)

            # 3. Loop and apply Jacobian
            dLdZ_2d = np.zeros_like(dLdA_2d)
            for i in range(A_2d.shape[0]):
                a = A_2d[i]
                J = np.diag(a) - np.outer(a, a)
                dLdZ_2d[i] = dLdA_2d[i] @ J

            # 4. Reshape back to moved shape
            dLdZ = dLdZ_2d.reshape(self.A.shape)

            # 5. Move axis back to original position
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)

            return dLdZ



        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A  # Already reshaped version of self.A (used in earlier step)
            dLdZ = np.moveaxis(dLdZ_2d.reshape(A_moved.shape), -1, self.dim)  # Move softmax dim back to original position

        return dLdZ
 

    