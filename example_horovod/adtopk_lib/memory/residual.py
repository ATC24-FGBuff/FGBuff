from adtopk_lib import Memory


class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        # numel, shape = ctx
        # values, indices = tensor_compressed
        # if values.numel()!=numel:
        #     tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        # else:
        #     tensor_decompressed=values
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx, name)

        residual = tensor - tensor_decompressed
        self.residuals[name] = residual
