import typing

import torch

import configs


class HardenedIdentity(torch.nn.Identity):
    __TO_NAN_VALUE = 0.0

    # values from profiling, see profiler.py
    __PROFILES = {
        configs.VIT_BASE_PATCH16_224: (-35.710975646972656, 63.46443557739258),
        configs.VIT_BASE_PATCH16_384: (-55.64329147338867, 66.98040771484375),
        configs.VIT_BASE_PATCH32_224_SAM: (-32.792327880859375, 37.55152130126953),
        configs.VIT_LARGE_PATCH14_CLIP_224: (-231.32127380371094, 124.64649200439453),
        configs.VIT_HUGE_PATCH14_CLIP_224: (-83.42196655273438, 90.81693267822266),
        configs.SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K: (-18.612855911254883, 18.950904846191406),
        configs.SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K: (-18.642213821411133, 18.5057430267334),
        configs.SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K: (-22.530014038085938, 22.742937088012695),
        configs.SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K: (-22.32180404663086, 22.495521545410156),
        configs.EVA_BASE_PATCH14_448_MIM: (-703.5409545898438, 223.30084228515625),
        configs.EVA_LARGE_PATCH14_448_MIM: (-904.2136840820312, 483.6650390625),
        configs.EVA_SMALL_PATCH14_448_MIN: (-342.5953369140625, 327.6106262207031),
        configs.MAXVIT_LARGE_TF_384: (-66806.84375, 35259.44140625),
        configs.MAXVIT_LARGE_TF_512: (-83250.609375, 40806.88671875),
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
        return (
            torch.clamp(  # Remove too large values, positive and negative
                input=torch.nan_to_num(  # Remove all nan values
                    super(HardenedIdentity, self).forward(input=input),  # Ensure full compatibility
                    nan=self.__TO_NAN_VALUE,
                    posinf=self.__max_layer_value,
                    neginf=self.__min_layer_value,
                ),
                min=self.__min_layer_value,
                max=self.__max_layer_value,  # Other values that are too large and are not inf
            )
        )
