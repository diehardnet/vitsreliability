#!/usr/bin/python3

import torch

import configs
import hardened_identity
from hardened_identity import replace_identity
import GroundingDINO.groundingdino.datasets.transforms as gdino_transforms
from GroundingDINO.groundingdino.util.misc import collate_fn as gdino_collate_fn
from GroundingDINO.groundingdino.util import get_tokenlizer as gdino_get_tokenlizer
from GroundingDINO.groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from GroundingDINO.groundingdino.models import build_model as gdino_build_model
from GroundingDINO.groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict

from GroundingDINO.demo.test_ap_on_coco import CocoDetection as GDINOCocoDetection
from GroundingDINO.demo.test_ap_on_coco import PostProcessCocoGrounding as GDINOPostProcessCocoGrounding
from GroundingDINO.groundingdino.datasets.cocogrounding_eval import (
    CocoGroundingEvaluator as GDINOCocoGroundingEvaluator
)

BATCH_SIZE = 1
DEVICE = "cuda:0"
_MIN_IDENTITY_VALUES_PROFILE = list()
_MAX_IDENTITY_VALUES_PROFILE = list()
COCO_DATASET_DIR = "/home/COCO"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"


def load_model(precision, model_config_path, model_checkpoint_path):
    # The First option is the baseline option
    cfg_args = gdino_SLConfig.fromfile(model_config_path)
    cfg_args.device = configs.GPU_DEVICE
    model = gdino_build_model(cfg_args)
    checkpoint = torch.load(model_checkpoint_path, map_location=configs.GPU_DEVICE)
    model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    # Disable also parameter grads
    model.zero_grad(set_to_none=True)
    model = model.to(configs.GPU_DEVICE)

    if precision == configs.FP16:
        model = model.half()

    return cfg_args.text_encoder_type, model


@torch.no_grad()
def main():
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        raise ValueError(f"Device {DEVICE} not available.")

    print("Creating model")
    text_encoder_type, model = load_model(
        model_checkpoint_path="/home/carol/vitsreliability/data/weights_grounding_dino/groundingdino_swinb_cogcoor.pth",
        model_config_path="/home/carol/vitsreliability/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        precision="fp32"
    )
    replace_identity(module=model, profile_or_inference="profile")
    print("Model created, creating the test loader")
    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = GDINOCocoDetection(COCO_DATASET_VAL, COCO_DATASET_ANNOTATIONS, transforms=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                              collate_fn=gdino_collate_fn)
    coco_api = dataset.coco

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    # TODO: try a different prompt
    print(f"Input text prompt:{caption}")
    # For each image in the batch I'm searching for the COCO classes
    input_captions = [caption]

    # build post processor
    tokenlizer = gdino_get_tokenlizer.get_tokenlizer(text_encoder_type)
    postprocessor = GDINOPostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = GDINOCocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    print("Test loader created, iterating over the dataset")

    for it, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs, captions=input_captions)
        torch.cuda.synchronize(device=DEVICE)
        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(inputs.device)
        results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)
        print(f"Iteration {it}")
        # if it == 10: break

    min_val, max_val = hardened_identity.get_min_max_profiled_values()
    print(f"MAX Value found:{min_val}")
    print(f"MIN Value found:{max_val}")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == '__main__':
    main()
