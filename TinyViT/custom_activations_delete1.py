"""
This file consists of custom activation functions
"""

from typing import Union, List, Tuple

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU


class GoLU_clamp(nn.Module):
    
    def __init__(self, inplace: bool = False):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):

        return x * torch.exp(-torch.exp(-torch.clamp(x, max=torch.tensor(10.0).to(x.device), min=-torch.tensor(10.0).to(x.device))))
    
# class GoLU_clamp(nn.Module):
    
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x * torch.exp(-torch.exp(-torch.clamp(x, max=torch.tensor(10.0).to(x.device), min=-torch.tensor(10.0).to(x.device))))
    

@torch.jit.script
def _golu_forward(
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter
) -> torch.Tensor:
    
    # Return the forward pass of GoLU Activation
    return z * alpha * torch.exp(-beta * torch.exp(-gamma * z))


@torch.jit.script
def _golu_backward(
    grad_output: torch.Tensor,
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter,
    req_grad: Tuple[bool, bool, bool, bool]
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    
    # Computes gradients for the backward pass
    if req_grad[0]:
        grad_z = grad_output * alpha * torch.exp(-beta * torch.exp(-gamma * z)) * (1 + z * beta * gamma * torch.exp(-gamma * z))
    else:
        grad_z = None
    
    if req_grad[1]:
        grad_alpha = grad_output * z * torch.exp(-beta * torch.exp(-gamma * z))
    else:
        grad_alpha = None
    
    if req_grad[2]:
        grad_beta = grad_output * -z * alpha * torch.exp(-beta * torch.exp(-gamma * z)) * torch.exp(-gamma * z)
    else:
        grad_beta = None
    
    if req_grad[3]:
        grad_gamma = grad_output * z * z * alpha * beta * torch.exp(-beta * torch.exp(-gamma * z)) * torch.exp(-gamma * z)
    else:
        grad_gamma = None
    
    # Return the gradients computed as per the backward pass
    return grad_z, grad_alpha, grad_beta, grad_gamma


@torch.jit.script
def _golu_forward_tanh(
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter
) -> torch.Tensor:
    
    # Return the forward pass of GoLU Activation
    return z * alpha * torch.sqrt(
        (1 + torch.tanh(-beta * torch.exp(-gamma * z))) / (1 - torch.tanh(-beta * torch.exp(-gamma * z)))
    )


@torch.jit.script
def _golu_backward_tanh(
    grad_output: torch.Tensor,
    z: torch.Tensor,
    alpha: nn.Parameter,
    beta: nn.Parameter,
    gamma: nn.Parameter,
    req_grad: Tuple[bool, bool, bool, bool]
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    
    # Computes gradients for the backward pass
    if req_grad[0]:
        grad_z = grad_output * alpha * (1 + z * beta * gamma * torch.exp(-gamma * z)) * torch.sqrt(
        (1 + torch.tanh(-beta * torch.exp(-gamma * z))) / (1 - torch.tanh(-beta * torch.exp(-gamma * z)))
    )
    else:
        grad_z = None
    
    if req_grad[1]:
        grad_alpha = grad_output * z * torch.sqrt(
        (1 + torch.tanh(-beta * torch.exp(-gamma * z))) / (1 - torch.tanh(-beta * torch.exp(-gamma * z)))
    )
    else:
        grad_alpha = None
    
    if req_grad[2]:
        grad_beta = grad_output * -z * alpha * torch.exp(-gamma * z) * torch.sqrt(
        (1 + torch.tanh(-beta * torch.exp(-gamma * z))) / (1 - torch.tanh(-beta * torch.exp(-gamma * z)))
    )
    else:
        grad_beta = None
    
    if req_grad[3]:
        grad_gamma = grad_output * z * z * alpha * beta * torch.exp(-gamma * z) * torch.sqrt(
        (1 + torch.tanh(-beta * torch.exp(-gamma * z))) / (1 - torch.tanh(-beta * torch.exp(-gamma * z)))
    )
    else:
        grad_gamma = None
    
    # Return the gradients computed as per the backward pass
    return grad_z, grad_alpha, grad_beta, grad_gamma


class GoLUFunction(Function):
    
    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        alpha: nn.Parameter,
        beta: nn.Parameter,
        gamma: nn.Parameter
    ) -> torch.Tensor:
        ctx.save_for_backward(z, alpha, beta, gamma)
        return _golu_forward(z=z, alpha=alpha, beta=beta, gamma=gamma)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        z, alpha, beta, gamma = ctx.saved_tensors  # Gives the saved tensors during forward pass
        req_grad = ctx.needs_input_grad
        return _golu_backward(grad_output=grad_output, z=z, alpha=alpha, beta=beta, gamma=gamma, req_grad=req_grad)


class GoLUFunctionTanh(Function):
    
    @staticmethod
    def forward(
        ctx,
        z: torch.Tensor,
        alpha: nn.Parameter,
        beta: nn.Parameter,
        gamma: nn.Parameter
    ) -> torch.Tensor:
        ctx.save_for_backward(z, alpha, beta, gamma)
        return _golu_forward_tanh(z=z, alpha=alpha, beta=beta, gamma=gamma)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        z, alpha, beta, gamma = ctx.saved_tensors  # Gives the saved tensors during forward pass
        req_grad = ctx.needs_input_grad
        return _golu_backward_tanh(grad_output=grad_output, z=z, alpha=alpha, beta=beta, gamma=gamma, req_grad=req_grad)


class GoLU(nn.Module):
    """
    GoLU Activation Function
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        approximator: str = 'exp',
        requires_grad: List = [False, False, False],
        clamp_alpha: List = [],
        clamp_beta: List = [],
        clamp_gamma: List = [],
    ) -> None:
        """
        Initialize the GoLU Activation Layer

        Gompertz Linear Unit (GoLU) is an element-wise and self-gated activation function. It uses the 
        Gompertz Function as a gate. The GoLU activation and Gompertz function are defined as follows,

        GoLU(x) = x * gompertz(x), where
        gompertz(x) = alpha * exp(-beta * exp(-gamma * x))

        The stationary activation has the respective alpha, beta and gamma values set to 1.0

        Args:
            alpha (float): Controls the upper asymptote or scale of the gate
            beta (float): Controls the gate displacement along x-axis
            gamma (float): Controls the growth rate of the gate
            approximator (str, optional): Option to choose the type of approximator. Default is 'exp'
            requires_grad (List): A list which decides if alpha, beta and gamma need to be trained. You can
            customize training the gate parameters.
            clamp_alpha (List): Clamping thresholds for alpha
            clamp_beta (List): Clamping thresholds for beta
            clamp_gamma (List): Clamping thresholds for gamma
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=requires_grad[0]), requires_grad=requires_grad[0])
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad=requires_grad[1]), requires_grad=requires_grad[1])
        self.gamma = nn.Parameter(torch.tensor(gamma, requires_grad=requires_grad[2]), requires_grad=requires_grad[2])
        self.approximator = approximator
        self.clamp_alpha = clamp_alpha
        self.clamp_beta = clamp_beta
        self.clamp_gamma = clamp_gamma

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls the forward pass of the GoLU Activation Function

        Returns:
            torch.Tensor: The output of the forward method
        """
        self.clamp_params_()
        return self.forward(*args)
    
    def clamp_params_(self) -> None:
        """
        This function clamps the parameters of the Gompertz Function
        """
        if self.clamp_alpha:
            assert (len(self.clamp_alpha) == 1) or (len(self.clamp_alpha) == 2)
            if len(self.clamp_alpha) == 1:
                self.alpha.data.clamp_(min=self.clamp_alpha[0])
            else:
                self.alpha.data.clamp_(min=self.clamp_alpha[0], max=self.clamp_alpha[1])

        if self.clamp_beta:
            assert (len(self.clamp_beta) == 1) or (len(self.clamp_beta) == 2)
            if len(self.clamp_beta) == 1:
                self.beta.data.clamp_(min=self.clamp_beta[0])
            else:
                self.beta.data.clamp_(min=self.clamp_beta[0], max=self.clamp_beta[1])

        if self.clamp_gamma:
            assert (len(self.clamp_gamma) == 1) or (len(self.clamp_gamma) == 2)
            if len(self.clamp_gamma) == 1:
                self.gamma.data.clamp_(min=self.clamp_gamma[0])
            else:
                self.gamma.data.clamp_(min=self.clamp_gamma[0], max=self.clamp_gamma[1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GoLU Activation Function

        Args:
            z (torch.Tensor): The input tensor on which the activation is computed

        Returns:
            torch.Tensor: The activated tensor
        """
        if self.approximator == 'tanh':
            return GoLUFunctionTanh.apply(z, self.alpha, self.beta, self.gamma)
        else:
            return GoLUFunction.apply(z, self.alpha, self.beta, self.gamma)


def get_activation_function(
    activation: str = 'ReLU'
) -> Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, GoLU]:
    """
    This function returns the activation needed to train an architecture

    Args:
        activation (str, optional): The name of the activation to train with. Defaults to 'ReLU'.

    Returns:
        Union[Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, GELU, SiLU, GoLU]: The activation function
    """
    if activation == 'Sigmoid':
        return Sigmoid()
    elif activation == 'Tanh':
        return Tanh()
    elif activation == 'ReLU':
        return ReLU()
    elif activation == 'Softplus':
        return Softplus()
    elif activation == 'LeakyReLU':
        return LeakyReLU()
    elif activation == 'PReLU':
        return PReLU()
    elif activation == 'ELU':
        return ELU()
    elif activation == 'SELU':
        return SELU()
    elif activation == 'GELU':
        return GELU()
    elif activation == 'Swish':
        return SiLU()
    elif activation == 'GoLU':
        return GoLU()
    elif activation == 'GoLU_alpha':
        return GoLU(
            requires_grad = [True, False, False], clamp_alpha=[]
        )
    elif activation == 'GoLU_beta':
        return GoLU(
            requires_grad = [False, True, False], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_gamma':
        return GoLU(
            requires_grad = [False, False, True], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_alpha_beta':
        return GoLU(
            requires_grad = [True, True, False], clamp_alpha=[], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_alpha_gamma':
        return GoLU(
            requires_grad = [True, False, True], clamp_alpha=[], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_beta_gamma':
        return GoLU(
            requires_grad = [False, True, True], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_alpha_beta_gamma':
        return GoLU(
            requires_grad = [True, True, True], clamp_alpha=[], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_tanh':
        return GoLU(
            approximator='tanh'
        )
    elif activation == 'GoLU_tanh_alpha':
        return GoLU(
            approximator='tanh', requires_grad = [True, False, False], clamp_alpha=[]
        )
    elif activation == 'GoLU_tanh_beta':
        return GoLU(
            approximator='tanh', requires_grad = [False, True, False], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_tanh_gamma':
        return GoLU(
            approximator='tanh', requires_grad = [False, False, True], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_tanh_alpha_beta':
        return GoLU(
            approximator='tanh', requires_grad = [True, True, False], clamp_alpha=[], clamp_beta=[0.02]
        )
    elif activation == 'GoLU_tanh_alpha_gamma':
        return GoLU(
            approximator='tanh', requires_grad = [True, False, True], clamp_alpha=[], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_tanh_beta_gamma':
        return GoLU(
            approximator='tanh', requires_grad = [False, True, True], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_tanh_alpha_beta_gamma':
        return GoLU(
            approximator='tanh', requires_grad = [True, True, True], clamp_alpha=[], clamp_beta=[0.02], clamp_gamma=[0.3, 1.0]
        )
    elif activation == 'GoLU_clamp':
        return GoLU_clamp()
    else:
        return ReLU()


def replace_ac_function(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_ac_function(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)