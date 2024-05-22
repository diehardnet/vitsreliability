import typing

import torch

import configs

_MIN_IDENTITY_VALUES_PROFILE, _MAX_IDENTITY_VALUES_PROFILE = list(), list()


class HardenedIdentity(torch.nn.Identity):
    __REPLACE_CORRUPTED_VALUES_BY = 0.0

    # values from profiling, see profiler.py
    __PROFILES = {
        configs.GROUNDING_DINO_SWINT_OGC: (-279.20074462890625, 500.12542724609375),
        configs.GROUNDING_DINO_SWINB_COGCOOR: (-361.2831115722656, 436.7012939453125),
        configs.VIT_BASE_PATCH16_224: (-35.710975646972656, 63.46443557739258),
        configs.VIT_BASE_PATCH16_384: (-55.64329147338867, 66.98040771484375),
        configs.SWIN_BASE_PATCH4_WINDOW7_224: (-5.720342636108398, 9.734111785888672),
        configs.SWIN_BASE_PATCH4_WINDOW12_384: (-5.822690963745117, 10.016140937805176),
        configs.DEIT_BASE_PATCH16_224: (-151.42042541503906, 79.11274719238281),
        configs.DEIT_BASE_PATCH16_384: (-176.51998901367188, 121.48692321777344),
    }

    # each min/max value is multiply by this constant to avoid too tight value restriction
    __BOUND_RATIO = 1.3

    def __init__(self, model_name: typing.Optional[str], *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        if model_name not in self.__PROFILES:
            raise NotImplementedError("The bounds of each model must be hardcoded.")
        # Set the bounds of values
        self.__min_layer_value, self.__max_layer_value = self.__PROFILES[model_name]
        self.__min_layer_value = round(self.__min_layer_value * self.__BOUND_RATIO, 0)
        self.__max_layer_value = round(self.__max_layer_value * self.__BOUND_RATIO, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior and implement smart hardening
        out = torch.nan_to_num(  # Remove all nan values
            super(HardenedIdentity, self).forward(input=input),  # Ensure full compatibility
            nan=self.__REPLACE_CORRUPTED_VALUES_BY,
            posinf=self.__REPLACE_CORRUPTED_VALUES_BY,
            neginf=self.__REPLACE_CORRUPTED_VALUES_BY,
        )
        out[(out < self.__min_layer_value) | (out > self.__max_layer_value)] = self.__REPLACE_CORRUPTED_VALUES_BY
        return out


class ProfileIdentity(torch.nn.Identity):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior and implement smart hardening
        global _MIN_IDENTITY_VALUES_PROFILE, _MAX_IDENTITY_VALUES_PROFILE
        _MIN_IDENTITY_VALUES_PROFILE.append(float(torch.min(input)))
        _MAX_IDENTITY_VALUES_PROFILE.append(float(torch.max(input)))
        return input


def replace_identity(module: torch.nn.Module, profile_or_inference: str, model_name: str = None) -> None:
    """Recursively put desired module in nn.module module."""

    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, torch.nn.Identity):
            if profile_or_inference == "profile":
                setattr(module, attr_str, ProfileIdentity())
            elif profile_or_inference == "inference":
                assert model_name is not None, "For inference you must specify the model name"
                setattr(module, attr_str, HardenedIdentity(model_name=model_name))
            else:
                raise AttributeError("Incorrect option, only 'profile' or 'inference' is allowed")

    # Iterate through immediate child modules. Note, our code does the recursion no need to use named_modules()
    for _, immediate_child_module in module.named_children():
        replace_identity(module=immediate_child_module, profile_or_inference=profile_or_inference,
                         model_name=model_name)


def get_min_max_profiled_values() -> [float, float]:
    return min(_MIN_IDENTITY_VALUES_PROFILE), max(_MAX_IDENTITY_VALUES_PROFILE)
