import torch
import torch.nn as nn

class LRP:
    """
    Layer-wise Relevance Propagation for PyTorch models.
    Supports Conv1D and Linear layers using epsilon and z+ rules.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1e-6):
        self.model = model
        self.epsilon = epsilon
        self.handles = []
        self.activations = {}
        self.register_hooks()

    def register_hooks(self):
        """
        Registers forward hooks to store activations.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                handle = module.register_forward_hook(self._store_activation(name))
                self.handles.append(handle)

    def _store_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def forward(self, x: torch.Tensor):
        """
        Forward pass to populate activations.
        """
        return self.model(x)

    def compute_lrp(self, x: torch.Tensor, class_idx: int = None, rule: str = "epsilon"):
        """
        Compute relevance scores for input x.
        x: (batch, C, T)
        class_idx: index of target class
        rule: "epsilon" or "zplus"
        Returns: relevance map of shape x
        """
        device = x.device
        self.model.eval()
        output = self.forward(x)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        if class_idx is None:
            # Max predicted class
            class_idx = logits.argmax(dim=1)
        # Initialize relevance
        R = torch.zeros_like(logits).to(device)
        for b in range(logits.shape[0]):
            R[b, class_idx[b]] = logits[b, class_idx[b]]
        # Backpropagate relevance
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv1d):
                R = self._lrp_conv1d(module, self.activations[name], R, rule)
            elif isinstance(module, nn.Linear):
                R = self._lrp_linear(module, self.activations[name], R, rule)
        # R now corresponds to input relevance
        return R

    def _lrp_conv1d(self, layer, A, R, rule):
        """
        LRP for Conv1D layer.
        A: input activations to conv layer (batch, C_in, T_in)
        R: relevance from output (batch, C_out, T_out)
        """
        W = layer.weight
        B = layer.bias if layer.bias is not None else torch.zeros(W.size(0)).to(W.device)
        Z = self._conv1d_forward(A, W, B) + (self.epsilon * torch.sign(R) if rule == "epsilon" else 0)
        S = R / Z
        # distribute relevance
        Z_T = self._conv1d_transpose(S, W, A.shape)
        return Z_T

    def _lrp_linear(self, layer, A, R, rule):
        """
        LRP for Linear layer.
        A: input activations (batch, features)
        R: relevance from output (batch, features_out)
        """
        W = layer.weight
        B = layer.bias if layer.bias is not None else torch.zeros(W.size(0)).to(W.device)
        Z = A @ W.t() + B + (self.epsilon * torch.sign(R) if rule == "epsilon" else 0)
        S = R / Z
        R_input = S @ W
        return R_input

    def _conv1d_forward(self, x, W, B):
        """
        Standard conv1d forward with same padding.
        """
        return nn.functional.conv1d(x, W, bias=B, stride=1, padding=W.shape[2]//2)

    def _conv1d_transpose(self, R, W, input_shape):
        """
        Distribute relevance back through Conv1D using transposed conv.
        """
        return nn.functional.conv_transpose1d(R, W, stride=1, padding=W.shape[2]//2, output_padding=0)[:, :, :input_shape[2]]
