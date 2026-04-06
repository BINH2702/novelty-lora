# Baseline
from methods.baseline import Baseline
from methods.inflora import InfLoRA     # https://arxiv.org/pdf/2404.00228
from methods.sdlora import SDLoRA       # https://arxiv.org/pdf/2501.13198
from methods.cllora import CLLoRA       # https://arxiv.org/pdf/2505.24816
from methods.ewclora import EWCLoRA
from methods.novelty_lora import NoveltyLoRA
from methods.directional_lora import DirectionalLoRA
from methods.dir_fis_lora import DirFisLoRA


def get_model(method, args):
    name = method.lower()
    options = {'baseline': Baseline,
               'inflora': InfLoRA,
               'sdlora': SDLoRA,
               'cllora': CLLoRA,
               'ewclora': EWCLoRA,
               'novelty_lora': NoveltyLoRA,
               'directional_lora': DirectionalLoRA,
               'dir_fis_lora': DirFisLoRA,
               'dfsc_ia': DirFisLoRA,
               'dfsc_ia_no_ia': DirFisLoRA,
               'dfsc_ia_no_adaptive_alpha': DirFisLoRA,
               'dfsc_ia_no_transport': DirFisLoRA,
               }
    return options[name](args)

