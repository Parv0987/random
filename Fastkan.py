import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable

class SplineLinear(nn.Linear):


    def __init__(self,in_features: int,out_features: int,init_scale: float = 0.1,**kwargs):
        self.init_scale = init_scale

        super().__init__(in_features=in_features,out_features=out_features,bias=False,**kwargs)

    def reset_parameters(self):

        nn.init.trunc_normal_(self.weight,mean=0.0,std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(self,grid_min: float = -2.0,grid_max: float = 2.0,num_grids: int = 8,denominator: Optional[float] = None):
        super().__init__()
        if num_grids < 2:
            raise ValueError("num_grids must be at least 2.")

        grid = torch.linspace(grid_min,grid_max,num_grids)

        self.register_buffer("grid",grid)

        if denominator is None:
            denominator = (grid_max - grid_min)/(num_grids - 1)

        if denominator <= 0:
            raise ValueError("denominator must be greater than zero.")

        self.denominator = denominator

    def forward(self,x: torch.Tensor):

        return torch.exp(
            -(
                ( x.unsqueeze(-1) - self.grid
                )/ self.denominator
            ) ** 2
        )
class Fast_KANLinear(nn.Module):
    def __init__(self,input_dim: int,output_dim: int,grid_min: float = -2.0,grid_max: float = 2.0,num_grids: int = 8,use_base_update: bool = True,base_activation: Callable = F.silu,spline_weight_init_scale: float = 0.1 ,spectral_response_init: float = 1.0):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.use_base_update = use_base_update
        self.base_activation = base_activation

        self.layernorm = nn.LayerNorm(input_dim)

        self.rbf = RadialBasisFunction(grid_min=grid_min,grid_max=grid_max,num_grids=num_grids)

        self.spline_linear = SplineLinear(in_features=input_dim * num_grids,out_features=output_dim,init_scale=spline_weight_init_scale)

        if use_base_update:

            self.base_linear = nn.Linear(input_dim,output_dim)
        else:self.base_linear = None
        self.spectral_response =nn.Parameter(torch.ones(input_dim)*spectral_response_init)

    def forward(self,x: torch.Tensor,time_benchmark: bool = False):

        if time_benchmark:
            normalized = x
        else:
            normalized = self.layernorm(x)

        spline_basis = self.rbf(normalized)
        num_grids = self.rbf.grid.numel()
        spline_weights =self.spline_linear.weight.reshape(self.output_dim ,self.input_dim , num_grids)
        spline_contribution = (spline_basis.unsqueeze(1)*spline_weights.unsqueeze(0)).sum(dim=-1)

        if self.use_base_update:

            base_input=(self.base_activation(x))
            base_contribution = (base_input.unsqueeze(1)* self.base_linear.weight.unsqueeze(0))
            contribution = (spline_contribution + base_contribution)
        else :
            contribution = spline_contribution

        spectral_response =F.softplus(self.spectral_response)
        spectral_response =(spectral_response/(spectral_response.mean()+ 1e-8))
        contribution = (contribution * spectral_response.view(1 ,1 ,-1))
        output =contribution.sum(dim=-1)


        return output
class FastKAN(nn.Module):

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation: Callable = F.silu,
        spline_weight_init_scale: float = 0.1,
        spectral_response_init: float = 1.0
    ):

        super().__init__()

        if len(layers_hidden) < 2:

            raise ValueError(
                "layers_hidden must contain at least two values."
            )

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(
            layers_hidden[:-1],
            layers_hidden[1:]
        ):

            self.layers.append(

                Fast_KANLinear(input_dim=in_dim,output_dim=out_dim, grid_min=grid_min,grid_max=grid_max,num_grids=num_grids,use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,spectral_response_init=spectral_response_init)

            )

    def forward(
        self,
        x: torch.Tensor
    ):

        for layer in self.layers:

            x = layer(x)

        return x