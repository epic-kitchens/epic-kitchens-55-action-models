# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Copyright MIT HAN Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notice of change:
# Modified by Will Price to support multiple output classification layers and `features()` and
# `logits()` methods.

import logging
from typing import Tuple, Union

import numpy as np
import pretrainedmodels
import torch
from torch import nn
from torch.nn.init import constant_, normal_
from torch.utils import model_zoo

from ops.basic_ops import ConsensusModule
from pretrained_settings import urls as pretrained_urls
from pretrained_settings import InvalidPretrainError, ModelConfig

LOG = logging.getLogger(__name__)


class TSM(nn.Module):
    """
    Temporal Shift Module

    See https://arxiv.org/abs/1811.08383 for details.

    Args:
        num_class:
            Number of classes, can be either a single integer,
            or a 2-tuple for training verb+noun multi-task models
        num_segments:
            Number of frames/optical flow stacks input into the model
        modality:
            Either ``RGB`` or ``Flow``.
        base_model:
            Backbone model architecture one of ``resnet18``, ``resnet30``,
            ``resnet50``, ``BNInception``.
        new_length:
            The number of channel inputs per snippet
        consensus_type:
            The consensus function used to combined information across segments.
            One of ``avg``, ``max``, ``TRN``, ``TRNMultiscale``.
        before_softmax:
            Whether to output class score before or after softmax.
        dropout:
            The dropout probability. The dropout layer replaces the backbone's
            classification layer.
        img_feature_dim:
            Only for TRN/MTRN models. The dimensionality of the features used for
            relational reasoning.
        partial_bn:
            Whether to freeze all BN layers beyond the first 2 layers.
        shift_div:
            The reciprocal of the proportion of channels that will be shifted
            along the time dimension.
        shift_place:
            Either ``'blockres'`` or ``'block'``. The former will place the shift
            module in the residual branch (only compatible with ResNet derived
            backbones), and the latter will place the shift module on the main
            network path.
        fc_lr5:
            Whether to add a x5 multiplier to the the fully connected layer
        temporal_pool:
            Whether to gradually temporally pool throughout the network
        non_local:
            Whether to inject non-local blocks
        pretrained:
            Either ``'imagenet'`` for ImageNet initialised models,
            or ``'epic-kitchens'`` for weights pretrained on EPIC-Kitchens.
    """

    def __init__(
        self,
        num_class,
        num_segments,
        modality,
        base_model="resnet50",
        new_length=None,
        consensus_type="avg",
        before_softmax=True,
        dropout=0.8,
        partial_bn=True,
        shift_div=8,
        shift_place="blockres",
        fc_lr5=False,
        temporal_pool=False,
        non_local=False,
        pretrained="imagenet",
    ):
        super(TSM, self).__init__()
        self.arch = base_model
        self.num_class = num_class
        self.is_multitask = isinstance(num_class, (list, tuple))
        if self.is_multitask:
            assert len(self.num_class) == 2, (
                "We only support 2 tasks in multi task " "problems"
            )
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.pretrained = pretrained

        self.is_shift = True
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if not before_softmax and consensus_type != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        LOG.info(
            (
                """
Initializing TSM with base model: {}.
TSM Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(
                    base_model,
                    self.modality,
                    self.num_segments,
                    self.new_length,
                    consensus_type,
                    self.dropout,
                )
            )
        )

        self._prepare_base_model(base_model)

        self.feature_dim = self._prepare_tsn(num_class)

        if self.modality == "Flow":
            LOG.info("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            LOG.info("Done. Flow model ready...")
        elif self.modality == "RGBDiff":
            LOG.info("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            LOG.info("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
        if pretrained and pretrained != "imagenet":
            self._load_pretrained_model(pretrained)

    def _load_pretrained_model(self, pretrained):
        config = self._get_pretrained_model_config()
        try:
            weights_url = pretrained_urls[pretrained][config]
        except KeyError:
            raise InvalidPretrainError(
                "The model configuration {} has no pretrained checkpoint".format(config)
            )
        checkpoint_dict = model_zoo.load_url(weights_url)
        if checkpoint_dict["segment_count"] != self.num_segments:
            raise ValueError(
                "Checkpoint was trained with {} segments, but model is "
                "configured for {} segments.".format(
                    checkpoint_dict["segment_count"], self.num_segments
                )
            )
        if checkpoint_dict["modality"] != self.modality:
            raise ValueError(
                "Checkpoint is trained for {} input, but model is "
                "configured for {} input.".format(
                    checkpoint_dict["modality"], self.modality
                )
            )
        state_dict = checkpoint_dict["state_dict"]
        self.load_state_dict(state_dict)

    def _get_pretrained_model_config(self):
        return ModelConfig(
            variant="TSM",
            base_model=self.arch,
            modality=self.modality,
            num_segments=self.num_segments,
            consensus_type=self.consensus_type,
        )

    def _initialise_layer(self, layer, mean=0, std=0.001):
        normal_(layer.weight, mean, std)
        constant_(layer.bias, mean)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name
        ).in_features
        setattr(
            self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout)
        )
        if self.is_multitask:
            self.fc_verb = nn.Linear(feature_dim, num_class[0])
            self.fc_noun = nn.Linear(feature_dim, num_class[1])
            self._initialise_layer(self.fc_verb)
            self._initialise_layer(self.fc_noun)
        else:
            self.new_fc = nn.Linear(feature_dim, num_class)
            self._initialise_layer(self.new_fc)
        return feature_dim

    def _remove_last_layer(self):
        delattr(self.base_model, self.base_model.last_layer_name)
        for tup in self.base_model._op_list:
            if tup[0] == self.base_model.last_layer_name:
                self.base_model._op_list.remove(tup)

    def _prepare_base_model(self, base_model):
        LOG.info("=> base model: {}".format(base_model))

        backbone_pretrained = "imagenet" if self.pretrained == "imagenet" else None
        if "resnet" in base_model.lower():
            self.base_model = getattr(pretrainedmodels, base_model)(
                pretrained=backbone_pretrained
            )
            if self.is_shift:
                LOG.info("Adding temporal shift...")
                from ops.temporal_shift import make_temporal_shift

                make_temporal_shift(
                    self.base_model,
                    self.num_segments,
                    n_div=self.shift_div,
                    place=self.shift_place,
                    temporal_pool=self.temporal_pool,
                )

            if self.non_local:
                LOG.info("Adding non-local module...")
                from ops.non_local import make_non_local

                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = "last_linear"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == "Flow":
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == "RGBDiff":
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = (
                    self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
                )

        elif base_model.lower() == "bninception":
            from archs import bninception

            self.base_model = bninception(pretrained=backbone_pretrained)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = "fc"
            if self.modality == "Flow":
                self.input_mean = [128]
            elif self.modality == "RGBDiff":
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                LOG.info("Adding temporal shift...")
                self.base_model.build_temporal_ops(
                    self.num_segments,
                    is_temporal_shift=self.shift_place,
                    shift_div=self.shift_div,
                )
        else:
            raise ValueError("Unknown base model: {}".format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSM, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            LOG.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv3d)
            ):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy".format(
                            type(m)
                        )
                    )

        return [
            {
                "params": first_conv_weight,
                "lr_mult": 5 if self.modality == "Flow" else 1,
                "decay_mult": 1,
                "name": "first_conv_weight",
            },
            {
                "params": first_conv_bias,
                "lr_mult": 10 if self.modality == "Flow" else 2,
                "decay_mult": 0,
                "name": "first_conv_bias",
            },
            {
                "params": normal_weight,
                "lr_mult": 1,
                "decay_mult": 1,
                "name": "normal_weight",
            },
            {
                "params": normal_bias,
                "lr_mult": 2,
                "decay_mult": 0,
                "name": "normal_bias",
            },
            {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "BN scale/shift"},
            {"params": custom_ops, "lr_mult": 1, "decay_mult": 1, "name": "custom_ops"},
            # for fc
            {"params": lr5_weight, "lr_mult": 5, "decay_mult": 1, "name": "lr5_weight"},
            {"params": lr10_bias, "lr_mult": 10, "decay_mult": 0, "name": "lr10_bias"},
        ]

    def features(self, input: torch.Tensor) -> torch.Tensor:
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == "RGBDiff":
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_model_input = input.view((-1, sample_len) + input.size()[-2:])
        return self.base_model(base_model_input)

    def logits(
        self, features: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.is_multitask:
            logits_verb = self.fc_verb(features)
            logits_noun = self.fc_noun(features)

            if not self.before_softmax:
                logits_verb = self.softmax(logits_verb)
                logits_noun = self.softmax(logits_noun)

            if self.reshape:
                if self.is_shift and self.temporal_pool:
                    logits_verb = logits_verb.view(
                        (-1, self.num_segments // 2) + logits_verb.size()[1:]
                    )
                    logits_noun = logits_noun.view(
                        (-1, self.num_segments // 2) + logits_noun.size()[1:]
                    )
                else:
                    logits_verb = logits_verb.view(
                        (-1, self.num_segments) + logits_verb.size()[1:]
                    )
                    logits_noun = logits_noun.view(
                        (-1, self.num_segments) + logits_noun.size()[1:]
                    )
            output_verb = self.consensus(logits_verb)
            output_noun = self.consensus(logits_noun)
            return output_verb.squeeze(1), output_noun.squeeze(1)
        else:
            logits = self.new_fc(features)

            if not self.before_softmax:
                logits = self.softmax(logits)

            if self.reshape:
                if self.is_shift and self.temporal_pool:
                    logits = logits.view(
                        (-1, self.num_segments // 2) + logits.size()[1:]
                    )
                else:
                    logits = logits.view((-1, self.num_segments) + logits.size()[1:])
            output = self.consensus(logits)
            return output.squeeze(1)

    def forward(
        self, input: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        features = self.features(input)
        return self.logits(features)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view(
            (-1, self.num_segments, self.new_length + 1, input_c) + input.size()[2:]
        )
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = (
                    input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                )
            else:
                new_data[:, :, x - 1, :, :, :] = (
                    input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                )

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(
            filter(
                lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
            )
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = (
            params[0]
            .data.mean(dim=1, keepdim=True)
            .expand(new_kernel_size)
            .contiguous()
        )

        new_conv = nn.Conv2d(
            2 * self.new_length,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == "BNInception":
            import torch.utils.model_zoo as model_zoo

            sd = model_zoo.load_url(
                "https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1"
            )
            base_model.load_state_dict(sd)
            LOG.info("=> Loading pretrained Flow weight done...")
        else:
            LOG.warning("#" * 30 + "Warning! No Flow pretrained model is found")
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(
            lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))
        )[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = (
                params[0]
                .data.mean(dim=1, keepdim=True)
                .expand(new_kernel_size)
                .contiguous()
            )
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (
                    params[0].data,
                    params[0]
                    .data.mean(dim=1, keepdim=True)
                    .expand(new_kernel_size)
                    .contiguous(),
                ),
                1,
            )
            new_kernel_size = (
                kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]
            )

        new_conv = nn.Conv2d(
            new_kernel_size[1],
            conv_layer.out_channels,
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            bias=True if len(params) == 2 else False,
        )
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
