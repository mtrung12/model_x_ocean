from .log import log_to_file
from .parser import *
from .gpt_client import gpt_call


def hf_call(*args, **kwargs):
    from .hf_client import hf_call as _hf_call

    return _hf_call(*args, **kwargs)
