import sys
sys.path.append("/home/carol/vitsreliability/GroundingDINO")
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

GROUNDING_DINO_PATH = "/home/carol/vitsreliability/GroundingDINO"

model = load_model(f"{GROUNDING_DINO_PATH}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   f"{GROUNDING_DINO_PATH}/weights/groundingdino_swint_ogc.pth").to("cuda:0")
IMAGE_PATH = f"/home/COCO/val2017/000000012670.jpg"
TEXT_PROMPT = "all boys wearing red tshirt . "
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image.to("cuda:0"),
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("/tmp/annotated_image.jpg", annotated_frame)
