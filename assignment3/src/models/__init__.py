from fusion.early_fusion import ConcatFusion
from models.modality_a_encoder import ImageOnly
from models.modality_b_encoder import TextOnly

MODELS = {"image_only": ImageOnly, "text_only": TextOnly, "concat_fusion": ConcatFusion}
