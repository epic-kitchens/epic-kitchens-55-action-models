#!/usr/bin/env python
# Copyright (c) 2016, Multimedia Laboratory, The Chinese University of Hong Kong
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Notice of change:
# Modified by Will Price to support multiple output classification layers and `features()` and
# `logits()` methods.

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pretrainedmodels
import torch
from torch import nn
from torch.nn.init import constant_, normal_
from torch.utils import model_zoo

from pretrained_settings import urls as pretrained_urls, InvalidPretrainError
from pretrained_settings import ModelConfig
from ops.basic_ops import ConsensusModule
from ops.trn import return_TRN

LOG = logging.getLogger(__name__)


class TSN(nn.Module):
    """
    Temporal Segment Network

    See https://arxiv.org/abs/1608.00859 for more details.

    Args:
        num_class:
            number of classes, can be either a single integer,
            or a 2-tuple for training verb+noun multi-task models
        num_segments:
            number of frames/optical flow stacks input into the model
        modality:
            either ``rgb`` or ``flow``.
        base_model:
            backbone model architecture one of ``resnet18``, ``resnet30``,
            ``resnet50``, ``bninception``, ``inceptionv3``, ``vgg16``.
            ``bninception`` and ``resnet50`` are the most thoroughly tested.
        new_length:
            the number of channel inputs per snippet
        consensus_type:
            the consensus function used to combined information across segments.
            one of ``avg``, ``max``, ``trn``, ``trnmultiscale``.
        before_softmax:
            whether to output class score before or after softmax.
        dropout:
            the dropout probability. the dropout layer replaces the backbone's
            classification layer.
        img_feature_dim:
            only for trn/mtrn models. the dimensionality of the features used for
            relational reasoning.
        partial_bn:
            whether to freeze all bn layers beyond the first 2 layers.
        pretrained:
            either ``'imagenet'`` for imagenet initialised models,
            or ``'epic-kitchens'`` for weights pretrained on epic-kitchens.
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
        dropout=0.7,
        img_feature_dim=256,
        partial_bn=True,
        pretrained="imagenet",
    ):

        super(TSN, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.modality = modality
        self.arch = base_model
        self.consensus_type = consensus_type
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.img_feature_dim = img_feature_dim
        self._enable_pbn = partial_bn
        self.pretrained = pretrained
        self.reshape = True
        if not before_softmax and consensus_type != "avg":
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        LOG.info(
            """
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    img_feature_dim:    {}
    dropout_ratio:      {}
        """.format(
                base_model,
                self.modality,
                self.num_segments,
                self.new_length,
                self.consensus_type,
                self.img_feature_dim,
                self.dropout,
            )
        )

        self._prepare_base_model(base_model)

        self.feature_dim = getattr(
            self.base_model, self.base_model.last_layer_name
        ).in_features
        self._prepare_tsn()

        if self.modality == "Flow":
            LOG.info("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            LOG.debug("Done. Flow model ready...")
        elif self.modality == "RGBDiff":
            LOG.info("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            LOG.debug("Done. RGBDiff model ready.")

        if consensus_type.startswith("TRN"):
            self.consensus = return_TRN(
                consensus_type, self.img_feature_dim, self.num_segments, num_class
            )
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

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
        if self.consensus_type == "TRN":
            variant = "TRN"
        elif self.consensus_type == "TRNMultiscale":
            variant = "MTRN"
        else:
            variant = "TSN"

        return ModelConfig(
            variant=variant,
            base_model=self.arch,
            modality=self.modality,
            num_segments=self.num_segments,
            consensus_type=self.consensus_type,
        )

    def _remove_last_layer(self):
        delattr(self.base_model, self.base_model.last_layer_name)
        for tup in self.base_model._op_list:
            if tup[0] == self.base_model.last_layer_name:
                self.base_model._op_list.remove(tup)

    def _initialise_layer(self, layer, mean=0, std=0.001):
        normal_(layer.weight, mean, std)
        constant_(layer.bias, mean)

    def _prepare_tsn(self):
        if self.consensus_type.startswith("TRN") or not isinstance(
            self.num_class, (list, tuple)
        ):
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(p=self.dropout),
            )
            if self.consensus_type.startswith("TRN"):
                self.new_fc = nn.Linear(self.feature_dim, self.img_feature_dim)
            else:
                self.new_fc = nn.Linear(self.feature_dim, self.num_class)
            self._initialise_layer(self.new_fc)
        else:
            assert (
                len(self.num_class) == 2
            ), "We only support 2 tasks in multi task problems"
            setattr(
                self.base_model,
                self.base_model.last_layer_name,
                nn.Dropout(p=self.dropout),
            )
            self.fc_verb = nn.Linear(self.feature_dim, self.num_class[0])
            self.fc_noun = nn.Linear(self.feature_dim, self.num_class[1])
            self._initialise_layer(self.fc_verb)
            self._initialise_layer(self.fc_noun)

    def _prepare_base_model(self, base_model):
        backbone_pretrained = "imagenet" if self.pretrained == "imagenet" else None

        if "resnet" in base_model.lower() or "vgg" in base_model.lower():
            self.base_model = getattr(pretrainedmodels, base_model)(
                pretrained=backbone_pretrained
            )
            self.base_model.last_layer_name = "last_linear"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == "Flow":
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == "RGBDiff":
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = (
                    self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
                )
        elif base_model.lower() == "bninception":
            self.base_model = getattr(pretrainedmodels, base_model.lower())(
                pretrained=backbone_pretrained
            )
            self.base_model.last_layer_name = "last_linear"
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == "Flow":
                self.input_mean = [128]
            elif self.modality == "RGBDiff":
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model.lower() == "inceptionv3":
            self.base_model = getattr(pretrainedmodels, base_model.lower())(
                pretrained=backbone_pretrained
            )
            self.base_model.last_layer_name = "top_cls_fc"
            self.input_size = 299
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == "Flow":
                self.input_mean = [128]
            elif self.modality == "RGBDiff":
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError("Unknown base model: {}".format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            LOG.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
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
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
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
        ]

    def features(self, input: torch.Tensor) -> torch.Tensor:
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == "RGBDiff":
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        return self.base_model.forward(input.view((-1, sample_len) + input.size()[-2:]))

    def logits(
        self, features: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(
            self.num_class, (list, tuple)
        ) and not self.consensus_type.startswith("TRN"):
            logits_verb = self.fc_verb(features)
            if not self.before_softmax:
                logits_verb = self.softmax(logits_verb)
            if self.reshape:
                logits_verb = logits_verb.view(
                    (-1, self.num_segments) + logits_verb.size()[1:]
                )
            output_verb = self.consensus(logits_verb)

            logits_noun = self.fc_noun(features)
            if not self.before_softmax:
                logits_noun = self.softmax(logits_noun)
            if self.reshape:
                logits_noun = logits_noun.view(
                    (-1, self.num_segments) + logits_noun.size()[1:]
                )
            output_noun = self.consensus(logits_noun)
            return output_verb.squeeze(1), output_noun.squeeze(1)
        else:
            # handle TRN model
            features = self.new_fc(features)
            features = features.view((-1, self.num_segments) + features.size()[1:])

            output = self.consensus(features)
            if isinstance(output, tuple):
                return tuple([o.squeeze(1) for o in output])

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
            .detach()
            .mean(dim=1, keepdim=True)
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
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][
            :-7
        ]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
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
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = (
                params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()
            )
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (
                    params[0].detach(),
                    params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous(),
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
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
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


def _vis_main(args):
    from torchviz import make_dot

    if args.class_type == "verb":
        class_count = args.verb_count
    elif args.class_type == "noun":
        class_count = args.noun_count
    elif args.class_type == "verb+noun":
        class_count = (args.verb_count, args.noun_count)
    else:
        raise ValueError("Unknown class type {}".format(args.class_type))

    if args.modality == "Flow":
        new_length = 5
        channel_count = 2
    elif args.modality == "RGB":
        new_length = 1
        channel_count = 3
    else:
        raise ValueError("Unknown modality {}".format(args.modality))

    model = TSN(
        class_count,
        args.segment_count,
        args.modality,
        args.base_model,
        new_length,
        consensus_type=args.consensus,
    )

    x = torch.randn((1, args.segment_count * channel_count, 224, 224))
    y = model(x)
    if isinstance(y, tuple):
        y = torch.cat(y, dim=-1)

    graph = make_dot(
        y, params=dict(list(model.named_parameters()) + [("x", x)])
    )  # type: graphviz.Graph
    print("Rendering graph to {}".format(args.arch_diagram))
    graph.render(args.arch_diagram, view=False, format=args.format)


if __name__ == "__main__":
    import configargparse
    import logging
    import sys

    parser = configargparse.ArgumentParser(
        description="", formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "arch_diagram", help="Path to save architecture diagram", type=Path
    )
    parser.add_argument("--segment-count", default=3, type=int)
    parser.add_argument(
        "--consensus", default="TRN", choices=["avg", "max", "TRN", "TRNmultiscale"]
    )
    parser.add_argument(
        "--class-type", default="verb+noun", choices=["verb+noun", "verb", "noun"]
    )
    parser.add_argument("--verb-count", default=124)
    parser.add_argument("--noun-count", default=352)
    parser.add_argument("--modality", default="RGB", choices=["RGB", "Flow"])
    parser.add_argument(
        "--base-model",
        default="BNInception",
        choices=["BNInception", "InceptionV3", "resnet101", "vgg16", "vgg19"],
    )
    parser.add_argument(
        "--format",
        default="pdf",
        help="Format to save graph in.These are 10 crop results.",
    )

    try:
        _vis_main(parser.parse_args())
    except ImportError:
        print("torchviz must be installed to generate an architecture diagram")
        sys.exit(1)


class TRN(TSN):
    """
    Single-scale Temporal Relational Network

    See https://arxiv.org/abs/1711.08496 for more details.
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
            ``resnet50``, ``BNInception``, ``InceptionV3``, ``VGG16``.
            ``BNInception`` and ``resnet50`` are the most thoroughly tested.
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
        before_softmax=True,
        dropout=0.7,
        img_feature_dim=256,
        partial_bn=True,
        pretrained="imagenet",
    ):

        super().__init__(
            num_class=num_class,
            num_segments=num_segments,
            modality=modality,
            base_model=base_model,
            new_length=new_length,
            consensus_type="TRN",
            before_softmax=before_softmax,
            dropout=dropout,
            img_feature_dim=img_feature_dim,
            partial_bn=partial_bn,
            pretrained=pretrained,
        )


class MTRN(TSN):
    """
    Multi-scale Temporal Relational Network

    See https://arxiv.org/abs/1711.08496 for more details.
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
            ``resnet50``, ``BNInception``, ``InceptionV3``, ``VGG16``.
            ``BNInception`` and ``resnet50`` are the most thoroughly tested.
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
        before_softmax=True,
        dropout=0.7,
        img_feature_dim=256,
        partial_bn=True,
        pretrained="imagenet",
    ):

        super().__init__(
            num_class=num_class,
            num_segments=num_segments,
            modality=modality,
            base_model=base_model,
            new_length=new_length,
            consensus_type="TRNMultiscale",
            before_softmax=before_softmax,
            dropout=dropout,
            img_feature_dim=img_feature_dim,
            partial_bn=partial_bn,
            pretrained=pretrained,
        )
