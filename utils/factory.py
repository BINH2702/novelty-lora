# Baseline
from methods.baseline import Baseline
from methods.inflora import InfLoRA     # https://arxiv.org/pdf/2404.00228
from methods.sdlora import SDLoRA       # https://arxiv.org/pdf/2501.13198
from methods.cllora import CLLoRA       # https://arxiv.org/pdf/2505.24816
from methods.ewclora import EWCLoRA
from methods.novelty_lora import NoveltyLoRA


def get_model(method, args):
    name = method.lower()
    options = {'baseline': Baseline,
               'inflora': InfLoRA,
               'sdlora': SDLoRA,
               'cllora': CLLoRA,
               'ewclora': EWCLoRA,
               'novelty_lora': NoveltyLoRA,
               }
    return options[name](args)

