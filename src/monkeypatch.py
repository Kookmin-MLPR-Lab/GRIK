"""Monkeypatch dispatch for GRIK.

Replaces ``transformers.LlamaAttention.forward`` and related hooks with the
KV-cache-aware variants in :mod:`src.llama_model`. This paper-submission
build registers a single method, ``grik``.
"""
from importlib.metadata import version
import warnings
import transformers

from src.llama_model import llama_attn_forward_H2O
from src.llama_model import llama_model_forward
from src.llama_model import prepare_inputs_for_generation_llama


# The only method exposed by this build.
#
#   grik = 32-step iterative channel scorer with Wanda K-norm prior and
#          GQA-mean aggregation through O_proj influence (α_h). See
#          src/kv_pruning_utils.py::key_pruner_iterative for the full
#          algorithm.
H2O_METHODS = ["grik"]

_TESTED_TRANSFORMERS = ("4.40", "4.41", "4.42", "4.43", "4.44", "4.45")


def replace_model(method, model_type="llama"):
    """Patch ``transformers.LlamaAttention.forward`` for GRIK.

    Args:
        method: must be ``"grik"`` (the only registered method).
        model_type: only ``"llama"`` is supported in this build.
    """
    transformers_version = version("transformers")
    if not any(transformers_version.startswith(v) for v in _TESTED_TRANSFORMERS):
        warnings.warn(
            f"Transformers version {transformers_version} is outside tested range "
            f"({_TESTED_TRANSFORMERS}); attention/forward hooks may need adjustments."
        )

    if method not in H2O_METHODS:
        raise ValueError(
            f"Unknown method: {method!r}. Only 'grik' is registered in this build."
        )

    print(f"[GRIK] Patching LLaMA attention with method={method}")
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama


def detect_model_type(model_path):
    """Return ``"llama"`` for any backbone in this paper's main table."""
    return "llama"


def replace_llama(method):
    """Backward-compat shim used by some scripts."""
    replace_model(method, model_type="llama")
