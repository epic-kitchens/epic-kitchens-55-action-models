from collections import namedtuple

__all__ = ["urls", "ModelConfig"]


ModelConfig = namedtuple(
    "ModelConfig",
    ["variant", "base_model", "modality", "num_segments", "consensus_type"],
)
_epic_url_base = (
    "https://data.bris.ac.uk/datasets/2tw6gdvmfj3f12papdy24flvmo/"
)


urls = {
    "epic-kitchens": {
        ModelConfig("TSN", "resnet50", "RGB", 8, "avg"): _epic_url_base
        + "TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar",
        ModelConfig("TSN", "resnet50", "Flow", 8, "avg"): _epic_url_base
        + "TSN_arch=resnet50_modality=Flow_segments=8-4317bc4a.pth.tar",
        ModelConfig("TSN", "BNInception", "RGB", 8, "avg"): _epic_url_base
        + "TSN_arch=BNInception_modality=RGB_segments=8-efb96e64.pth.tar",
        ModelConfig("TSN", "BNInception", "Flow", 8, "avg"): _epic_url_base
        + "TSN_arch=BNInception_modality=Flow_segments=8-4c720ee3.pth.tar",
        ModelConfig("TRN", "BNInception", "RGB", 8, "TRN"): _epic_url_base
        + "TRN_arch=BNInception_modality=RGB_segments=8-a770bfbd.pth.tar",
        ModelConfig("TRN", "BNInception", "Flow", 8, "TRN"): _epic_url_base
        + "TRN_arch=BNInception_modality=Flow_segments=8-4f84b178.pth.tar",
        ModelConfig("TRN", "resnet50", "RGB", 8, "TRN"): _epic_url_base
        + "TRN_arch=resnet50_modality=RGB_segments=8-c8176b38.pth.tar",
        ModelConfig("TRN", "resnet50", "Flow", 8, "TRN"): _epic_url_base
        + "TRN_arch=resnet50_modality=Flow_segments=8-c0a2821c.pth.tar",
        ModelConfig("MTRN", "BNInception", "RGB", 8, "TRNMultiscale"): _epic_url_base
        + "MTRN_arch=BNInception_modality=RGB_segments=8-8933f99e.pth.tar",
        ModelConfig("MTRN", "BNInception", "Flow", 8, "TRNMultiscale"): _epic_url_base
        + "MTRN_arch=BNInception_modality=Flow_segments=8-c0cea7e1.pth.tar",
        ModelConfig("MTRN", "resnet50", "RGB", 8, "TRNMultiscale"): _epic_url_base
        + "MTRN_arch=resnet50_modality=RGB_segments=8-46337796.pth.tar",
        ModelConfig("MTRN", "resnet50", "Flow", 8, "TRNMultiscale"): _epic_url_base
        + "MTRN_arch=resnet50_modality=Flow_segments=8-6667f285.pth.tar",
        ModelConfig("TSM", "resnet50", "RGB", 8, "avg"): _epic_url_base
        + "TSM_arch=resnet50_modality=RGB_segments=8-cfc93918.pth.tar",
        ModelConfig("TSM", "resnet50", "Flow", 8, "avg"): _epic_url_base
        + "TSM_arch=resnet50_modality=Flow_segments=8-e09c2d3a.pth.tar",
    }
}


class InvalidPretrainError(Exception):
    pass
