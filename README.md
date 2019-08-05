# EPIC-Kitchens' action recognition models

This is a set of models trained for EPIC-Kitchens baselines. We support:

- [TSN](https://github.com/yjxiong/tsn-pytorch)
- [TRN](https://github.com/metalbubble/TRN-pytorch)
- [TSM](https://github.com/MIT-HAN-LAB/temporal-shift-module)

Many thanks to the authors of these repositories.

## Set up

We provide an `environment.yml` file to create a conda environment. Sadly not all of the
set up can be encapsulated in this file, so you have to perform some steps yourself
(in the interest of eeking extra performance!)

```bash
$ conda env create -n epic-models -f environment.yml
$ conda activate epic-models

# The following steps are taken from
# https://docs.fast.ai/performance.html#installation

$ conda uninstall -y --force pillow pil jpeg libtiff
$ pip uninstall -y pillow pil jpeg libtiff
$ conda install -y -c conda-forge libjpeg-turbo
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
$ conda install -y jpeg libtiff
```

NOTE: If the installation of `pillow-simd` fails, you can try installing GCC from
conda-forge and trying the install again:

```bash
$ conda install -y gxx_linux-64
$ export CXX=x86_64-conda_cos6-linux-gnu-g++
$ export CC=x86_64-conda_cos6-linux-gnu-gcc
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
$ conda install -y jpeg libtiff
```

If you install any new packages, check that `pillow-simd` hasn't be overwritten
by an alternate `pillow` install by running:

```bash
$ python -c "from PIL import Image; print(Image.PILLOW_VERSION)"
```

You should see something like

```
6.0.0.post1
```

Pillow doesn't release with `post` suffixes, so if you have `post` in the version
name, it's likely you have `pillow-simd` installed.

## How to use the code

Check out `demo.py` for an example of how to construct the models and feed in
data, or read on below for how to load checkpointed models.

### Checkpoints

Checkpoints are saved as dictionaries with the following information:

- `model_type` (str): Variant. Either `'tsm'`, `'tsm-nl'`, `'tsn'`, `'trn'`, or
  `'mtrn'`
- `epoch` (int): Last epoch completed in training
- `segment_count` (int): Number of segments the network was trained with.
- `modality` (str): Modality of the input. Either `'RGB'` or `'Flow'`
- `state_dict` (dict): State dictionary of the network for use with
  `model.load_state_dict`
- `arch` (str): Modality of network. Either `'BNInception'` or `'resnet50'`.
- `args` (namespace): All the arguments used in training the network.

Some keys are only present depending on model type:
- TSN:
    - `consensus_type` (str, TSN only): Consensus module variant for TSN. Either `'avg'` or
      `'max'`.
- TSM:
    - `shift_place` (str, TSM only): Identifier for where the shift module is located.
      Either `block` or `blockres`.
    - `shift_div` (int, TSM only): The reciprocal of the proportion of channels used that
      are shifted.
    - `temporal_pool` (bool, TSM only): Whether gradual temporal pooling was used in this
      network.
    - `non_local` (bool, TSM only): Whether non-local blocks were added to this network.

To load checkpointed weights, first construct an instance of the network (using
information stored in the checkpoint about the architecture set up), then call
`model.load_state_dict`. For example:

```python
from tsn import TSN
import torch

verb_class_count, noun_class_count = 125, 352
class_count = (verb_class_count, noun_class_count)
ckpt = torch.load('TSN_modality=RGB_segments=8_arch=resnet50.pth')
model = TSN(
    num_class=class_count,
    num_segments=ckpt['segment_count'],
    modality=ckpt['modality'],
    base_model=ckpt['arch'],
    dropout=ckpt['args'].dropout
)
model.load_state_dict(ckpt['state_dict'])
```

We provide some helpers functions for this purpose in `model_loaders.py` so you can
simply load checkpoints like:

```python
from model_loader import load_checkpoint
model = load_checkpoint('path/to/checkpoint.pth.tar')
```


## Checkpoints
You can download checkpoints using the tool provided at `checkpoints/download.py`,
simply call it with the model variant, modality, and architecture that you wish to
download, e.g. `python checkpoints/download.py mtrn --arch BNInception --modality
Flow`. The checkpoint will be downloaded to the `checkpoints` directory.

The checkpoints accompanying this repository score the following on the test set
when using 10 crop evaluation.

| Checkpoint path                                          | Seen V@1  | Seen N@1  | Seen A@1  | Unseen V@1 | Unseen N@1 | Unseen A@1 |
|----------------------------------------------------------|-----------|-----------|-----------|------------|------------|------------|
| `TSN_arch=BNInception_modality=RGB_segments=8.pth.tar`   | 47.97     | 38.85     | 22.39     | 36.46      | 22.64      | 22.39      |
| `TSN_arch=BNInception_modality=Flow_segments=8.pth.tar`  | 51.68     | 26.82     | 16.76     | 47.35      | 21.20      | 13.49      |
| `TRN_arch=BNInception_modality=RGB_segments=8.pth.tar`   | 58.26     | 36.32     | 25.46     | 47.29      | 22.91      | 15.06      |
| `TRN_arch=BNInception_modality=Flow_segments=8.pth.tar`  | 55.20     | 23.95     | 16.03     | 50.32      | 19.02      | 12.77      |
| `MTRN_arch=BNInception_modality=RGB_segments=8.pth.tar`  | 55.76     | 37.94     | 26.62     | 45.41      | 23.90      | 15.57      |
| `MTRN_arch=BNInception_modality=Flow_segments=8.pth.tar` | 55.92     | 24.88     | 16.78     | 51.38      | 20.69      | 14.00      |
| `TSN_arch=resnet50_modality=RGB_segments=8.pth.tar`      | 49.71     | 39.85     | 23.97     | 36.70      | 23.11      | 12.77      |
| `TSN_arch=resnet50_modality=Flow_segments=8.pth.tar`     | 53.14     | 27.76     | 20.28     | 47.56      | 20.28      | 13.11      |
| `TRN_arch=resnet50_modality=RGB_segments=8.pth.tar`      | 58.82     | 37.27     | 26.62     | 47.32      | 23.69      | 15.71      |
| `TRN_arch=resnet50_modality=Flow_segments=8.pth.tar`     | 55.16     | 23.19     | 15.77     | 50.39      | 18.50      | 12.02      |
| `MTRN_arch=resnet50_modality=RGB_segments=8.pth.tar`     | **60.16** | 38.36     | **28.23** | 46.94      | **24.41**  | **16.32**  |
| `MTRN_arch=resnet50_modality=Flow_segments=8.pth.tar`    | 56.79     | 25.00     | 17.24     | 50.36      | 20.28      | 13.42      |
| `TSM_arch=resnet50_modality=RGB_segments=8.pth.tar`      | 57.88     | **40.84** | **28.22** | 43.50      | 23.32      | 14.99      |
| `TSM_arch=resnet50_modality=Flow_segments=8.pth.tar`     | 58.08     | 27.49     | 19.14     | **52.68**  | 20.83      | 14.27      |


## Extracting features

Both classes `TSN` and `TSM` include `features` and `logits` methods, mimicking the
[`pretrainedmodels`](https://github.com/Cadene/pretrained-models.pytorch) API. Simply
create a model instance `model = TSN(...)` and call `model.features(input)` to
obtain base-model features. To transform these to logits, call
`model.logits(features)` where `features` is the tensor obtained from the
previous step.

## Utilities
You can have a look inside the checkpoints using `python
tools/print_checkpoint_details.py <path-to-checkpoint>` to print checkpoint details
including the model variant, number of segments, modality, architecture, and weight
shapes.


## Citation

If you find our code and trained models helpful, please kindly cite our work
and dataset in addition to the authors of the models themselves (citation
information for this is in the following section).

```
@article{price2019_EvaluationActionRecognition,
    title={An Evaluation of Action Recognition Models on EPIC-Kitchens},
    author={Price, Will and Damen, Dima},
    journal={arXiv preprint arXiv:1908.00867},
    archivePrefix={arXiv},
    eprint={1908.00867},
    year={2019},
    month="Aug"
}
```

```
@inproceedings{damen2018_ScalingEgocentricVision,
   title={Scaling Egocentric Vision: The EPIC-KITCHENS Dataset},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and Fidler, Sanja and
           Furnari, Antonino and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan
           and Perrett, Toby and Price, Will and Wray, Michael},
   booktitle={European Conference on Computer Vision (ECCV)},
   year={2018}
}
```


## Acknowledgements

We'd like to thank the academics and authors responsible for the following codebases that enabled this work.

- [TSN](https://github.com/yjxiong/tsn-pytorch)
- [TRN](https://github.com/metalbubble/TRN-pytorch)
- [TSM](https://github.com/mit-han-lab/temporal-shift-module)
- [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

If you make use of this repository, please cite their work as well as ours

TSN:
```
@InProceedings{wang2016_TemporalSegmentNetworks,
    title={Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
    author={Limin Wang and Yuanjun Xiong and Zhe Wang and Yu Qiao and Dahua Lin and
            Xiaoou Tang and Luc {Val Gool}},
    booktitle={The European Conference on Computer Vision (ECCV)},
    year={2016}
}
```

TRN:
```
@InProceedings{zhou2017_TemporalRelationalReasoning,
    title={Temporal Relational Reasoning in Videos},
    author={Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    booktitle={The European Conference on Computer Vision (ECCV)},
    month={September},
    year={2018}
}
```

TSM:
```
@article{lin2018_TemporalShiftModule,
    title={Temporal Shift Module for Efficient Video Understanding},
    author={Lin, Ji and Gan, Chuang and Han, Song},
    journal={arXiv preprint arXiv:1811.08383},
    archivePrefix={arXiv},
    eprint={1811.08383},
    year={2018},
    month="Nov"
}
```
