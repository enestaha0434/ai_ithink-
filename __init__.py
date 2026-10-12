"""HyperAI'nin eğitimden bağımsız, yerel çekirdeği.

Bu paket hazır bir model çağırmaz. Model, tokenizer ve ağırlık yükleyicisi
eğitim hattı tarafından sağlanır; çekirdek yalnızca bunların birlikte güvenli
ve tutarlı biçimde çalışacağı sözleşmeyi tanımlar.
"""

from .contracts import GenerationResult, GenerationSettings
from .runtime import HyperAIRuntime
from .tokenizer import ByteTokenizer

__all__ = [
    "ByteTokenizer",
    "GenerationResult",
    "GenerationSettings",
    "HyperAILanguageModel",
    "HyperAIModelConfig",
    "HyperAIRuntime",
]


def __getattr__(name: str):
    """PyTorch yalnızca model sınıfı gerçekten istendiğinde içe aktarılır."""
    if name in {"HyperAILanguageModel", "HyperAIModelConfig"}:
        from .model import HyperAILanguageModel, HyperAIModelConfig

        return {"HyperAILanguageModel": HyperAILanguageModel, "HyperAIModelConfig": HyperAIModelConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
