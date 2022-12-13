import torch
from torch import nn
from inspect import signature
from typing import Type, Dict, Any

class Tower(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        repeats: int,
        static_block_args: Dict[str, Any] = None,
        dynamic_block_args: Dict[str, Any] = None,
        mults: Dict[str, float] = None
        ):
        """A tower of blocks.

        Parameters
        ----------
        block : Type[nn.Module]
        repeats : int
        static_block_args : Dict[str, Any]
            Arguments to initialize blocks that are static across repeats.
        dynamic_block_args : Dict[str, Any]
            Arguments to initialize blocks that change across repeats.
        mults : Dict[str, float]
            Multipliers for dynamic block arguments.
        """
        super().__init__()
        blocks = nn.ModuleList()
        if static_block_args is None:
            static_block_args = {}
        if dynamic_block_args is None:
            dynamic_block_args = {}
        if mults is None:
            mults = {}

        for arg, mult in mults.items():
            # replace initial value with geometric progression
            init_val = dynamic_block_args.get(arg, signature(block).parameters[arg].default)
            dynamic_block_args[arg] = (
                init_val*torch.logspace(
                    start=0,
                    end=repeats-1,
                    steps=repeats,
                    base=mult
                )
            ).to(dtype=signature(block).parameters[arg].annotation)

        for i in range(repeats):
            args = {arg: vals[i] for arg, vals in dynamic_block_args.items()}
            args.update(static_block_args)
            print(args)
            blocks.append(block(**args))
                
        self.tower = nn.Sequential(*blocks)

    def forward(self, x):
        return self.tower(x)
        