# <ins>MaxiM</ins>um corrupted v<ins>al</ins>ue<ins>s</ins> (MaxiMals)

MaxiMals is a set of Transformers hardened for reliability.

# Getting started

## Requirements
First, you have to have the following requirements:

- Python 3.10
- Python pip
- [ImageNet dataset](https://www.image-net.org/index.php)

### Reliability evaluation requirements

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
usage: setuppuretorch.py [-h] [--iterations ITERATIONS] [--testsamples TESTSAMPLES] [--generate] [--disableconsolelog] [--goldpath GOLDPATH] [--checkpointdir CHECKPOINTDIR] [--model MODEL] [--batchsize BATCHSIZE] [--usetorchcompile] [--hardenedid]

PyTorch Maximals radiation setup

options:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        Iterations to run forever
  --testsamples TESTSAMPLES
                        Test samples to be used in the test.
  --generate            Set this flag to generate the gold
  --disableconsolelog   Set this flag disable console logging
  --goldpath GOLDPATH   Path to the gold file
  --checkpointdir CHECKPOINTDIR
                        Path to checkpoint dir
  --model MODEL         Model name: resnet50d, tf_efficientnet_b7, vit_base_patch16_224, vit_base_patch32_224.sam, vit_base_patch16_384, vit_large_patch14_clip_224.laion2b_ft_in12k_in1k, vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k,
                        maxvit_large_tf_384.in21k_ft_in1k, maxvit_large_tf_512.in21k_ft_in1k, swinv2_large_window12to16_192to256.ms_in22k_ft_in1k, swinv2_large_window12to24_192to384.ms_in22k_ft_in1k, swinv2_base_window12to16_192to256.ms_in22k_ft_in1k,
                        swinv2_base_window12to24_192to384.ms_in22k_ft_in1k, eva02_large_patch14_448.mim_in22k_ft_in22k_in1k, eva02_base_patch14_448.mim_in22k_ft_in22k_in1k, eva02_small_patch14_336.mim_in22k_ft_in1k
  --batchsize BATCHSIZE
                        Batch size to be used.
  --usetorchcompile     Disable or enable torch compile (GPU Arch >= 700) <-- not currently working
  --hardenedid          Disable or enable HardenedIdentity. Work only for the profiled models.
```

For example, if you want to generate the golden file for the vit_base_patch32_224.sam model using 
the Identity layer hardening, with 40 Images from ImageNet divided into 4 batches, you can use the following command:

```shell
./setuppuretorch.py --testsamples 40 --batchsize 4 \ 
                    --hardenedid --generate \
                    --checkpointdir ./data/checkpoints \ 
                    --goldpath ./data/vit_base_patch32_224.sam_torch_compile_False_hardening_hardenedid.pt \
                    --model vit_base_patch32_224.sam
```

Then to run the same model for 20 iterations:

```shell
./setuppuretorch.py --testsamples 40 --batchsize 4 \ 
                    --hardenedid --iterations 20 \
                    --checkpointdir ./data/checkpoints \ 
                    --goldpath ./data/vit_base_patch32_224.sam_torch_compile_False_hardening_hardenedid.pt \
                    --model vit_base_patch32_224.sam
```
 
### Faults injections with NVBITFI

The **Nvidia Bit Fault Injector** allows to inject faults at the SASS level. We used it to inject faults on the tool we created to evaluate our technique.

The injector allows to inject faults for different instruction types and at different sites. We used a [modified version of NVBITFI](https://github.com/fernandoFernandeSantos/nvbitfi/tree/master) that allows us to inject faults on GPU warps.

We targeted the following sites : 
- **General Purpose Registers** : injecting *single bit flips* and *random values*
- **Load instructions** : injecting *single bit flips* and *random values*
- **32 bits Floating point instructions** : injecting *single bit flips*, *random values* and *random values on warps*


```python
# values for params.py script
inst_value_igid_bfm_map = {
    G_GP: [FLIP_SINGLE_BIT, RANDOM_VALUE], # General Purpose Registers
    G_LD: [FLIP_SINGLE_BIT, RANDOM_VALUE], # Load instructions
    G_FP32: [FLIP_SINGLE_BIT, RANDOM_VALUE, WARP_RANDOM_VALUE], # 32 bits Floating point instructions
}
```

#### ViTs profiling

To implement the technique, we needed to know what are the minimum and maximum values that pass through each ViT Identity layer on the whole Imagenet Validation set. In order to get the MIN and MAX values for each model, we perform an inference on each image of the dataset and save the MIN and MAX values for each model.

Once we had the values, we created the `HardenedIdentity` class that exhaustively applies the value range for the running model. If the value is not in the range, then the unwanted value is replaced by either MIN or MAX.


#### Sample Tool

In order to test our technique, [we created a tool to perform an inference on a batch of 5 images from the **ImageNet Validation Set**](https://github.com/lucasrqt/sample_tool). The tool allows us to perform multiple things. It works in two modes: **save** and **load**. 

The **save** mode will save the result of the inference to a file as a *gold save* to determine if later inferences are fault-free or not. 

The **load** will perform the inference on the same batch and will load the *gold save* file to compare the outputs of the inference. We check if the values in the output are different (Tolerable SDC) and also if the classification has changed (Critical SDC).

Adding to that, we can choose if the ViT model has to be in normal mode or hardened mode. By setting the *hardened* option, it will load the wanted ViT model and then replace all the Identity Layers with the hardened ones.



# Citation

To cite this work:

```bibtex
@inproceedings{roquet2023,
  TITLE = {{Cross-Layer Reliability Evaluation and
            Efficient Hardening of Large Vision Transformers Models}},
  AUTHOR = {Roquet, Lucas and Fernandes dos Santos, Fernando and Rech, Paolo
            and Traiola, Marcello and Sentieys, Olivier and Kritikakou, Angeliki},
  URL = {https://hal.science/hal-04456702},
  BOOKTITLE = {{Design Automation Conference (DAC)}},
  ADDRESS = {San Fracisco, United States},
  YEAR = {2024},
  MONTH = Jun,
  KEYWORDS = {Reliability ; Vision transformers ; GPU ; Radiation-induced effects},
  PDF = {https://hal.science/hal-04456702/file/main.pdf},
  HAL_ID = {hal-04456702},
  HAL_VERSION = {v1},
}
```

# Colaboration

If you encounter any issues with the code or feel that there is room for improvement,
please feel free to submit a pull request or open an issue.
