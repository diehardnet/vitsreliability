# Vision Transformers Reliability Assessment


# Getting started

## Requirements
First, you have to have the following requirements:

- Python 3.10
- Python pip
- [ImageNet dataset](https://www.image-net.org/index.php)

### Reliability evaluation requirements

numpy 1.24.1
torch 2.1.1+cu118
timm 0.9.12
transformers 4.41.0
 nvidia-ml-py
 addict
 yapf
 pycocotools
 

 Grouding DINO weights

 https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
 https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

 COCO validation dataset


For the fault simulations and beam experiments:

- For the beam experiments, you will need the scripts from [radhelper](https://github.com/radhelper) repositories 
to control the boards inside the beamline
  - You must have [libLogHelper](https://github.com/radhelper/libLogHelper) 
  installed on the host that controls the GPU and a socket server set up outside the beam room. 
  You can use [radiation-setup](https://github.com/radhelper/radiation-setup) as a socket server.
- For fault simulations, you can use the official version of 
[NVIDIA Bit Fault Injector](https://github.com/NVlabs/nvbitfi) (works until Volta micro-architecture) or 
the version
  we updated for [Ampere evaluations](https://github.com/fernandoFernandeSantos/nvbitfi/tree/new_gpus_support).


### Python libraries installation

Then install all the Python requirements.

```shell
python3 -m pip install -r requeriments.txt
```

## Executing the fault injection/radiation setup

```shell

```

For example, if you want to generate the golden file for the vit_base_patch32_224.sam model using 
the Identity layer hardening, with 40 Images from ImageNet divided into 4 batches, you can use the following command:

```shell

```

Then to run the same model for 20 iterations:

```shell

```
 
#### ViTs profiling

To implement the technique, we needed to know what are the minimum and maximum values that pass through each ViT Identity layer on the whole Imagenet Validation set. In order to get the MIN and MAX values for each model, we perform an inference on each image of the dataset and save the MIN and MAX values for each model.

Once we had the values, we created the `HardenedIdentity` class that exhaustively applies the value range for the running model. If the value is not in the range, then the unwanted value is replaced by either MIN or MAX.


# Acknowledgements
- grounding dino
- huggingface
- timm
- quantization git

# Citation

To cite this work:

```bibtex
```

# Collaboration

If you encounter any issues with the code or feel that there is room for improvement,
please feel free to submit a pull request or open an issue.
