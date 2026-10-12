"""Eğitilmiş yerel HyperAI checkpoint'i ile komut satırı sohbeti başlatır."""

from __future__ import annotations

import argparse
import re
import sys

from hyperai_core import GenerationSettings
from hyperai_core.data import format_chat_prompt
from hyperai_core.tokenizer import ByteTokenizer


def _print_sources(passages) -> None:
    """Arama belgelerini model cevabından ayrı, denetlenebilir biçimde gösterir."""
    if not passages:
        return
    print("\n[Bulunan kaynaklar — bunlar model talimatı değildir]")
    for index, passage in enumerate(passages, 1):
        excerpt = passage.text[:700] + ("…" if len(passage.text) > 700 else "")
        print(f"{index}. {passage.title} | {passage.source}\n   {excerpt}")


def _limit_sentences(text: str, maximum: int) -> str:
    """Küçük modelin doğru ilk cümleden sonra bozulmasını kullanıcıya taşımaz."""
    if maximum == 0:
        return text
    endings = list(re.finditer(r"[.!?](?:\s|$)", text))
    if len(endings) < maximum:
        return text
    return text[: endings[maximum - 1].end()].rstrip()


def main() -> None:
    # Windows'un eski konsol kod sayfalarında Türkçe/emoji yanıtların bozulmasını önler.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Kendi eğittiğiniz HyperAI modeliyle yerel sohbet.")
    parser.add_argument("--checkpoint", required=True, help="manifest.json içeren checkpoint klasörü")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Kısa sohbet için varsayılan üst sınır")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Tekrarı azaltmak için 1.0 ve üzeri değer")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="Tekrarlanması engellenecek token n-gram boyutu; 0 kapatır")
    parser.add_argument("--max-sentences", type=int, default=2, help="Gösterilecek en fazla cümle; 0 sınırsız")
    parser.add_argument("--knowledge-base", action="append", default=[], help="Yerel .txt/.md/.jsonl bilgi belgesi; birden çok kez verilebilir")
    parser.add_argument("--retrieval-top-k", type=int, default=3, help="Yerel bilgi tabanından gösterilecek sonuç sayısı")
    parser.add_argument("--wikipedia", action="store_true", help="Her soruda Türkçe Vikipedi'den en yakın sayfa özetini getirir")
    parser.add_argument("--web-timeout", type=float, default=8.0, help="Vikipedi isteği zaman aşımı (saniye)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Modelin çalışacağı cihaz")
    arguments = parser.parse_args()
    if arguments.max_sentences < 0:
        parser.error("--max-sentences negatif olamaz.")

    try:
        # PyTorch yalnızca gerçek model checkpoint'i başlatılırken gerekir.
        from hyperai_core.application import create_runtime
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            parser.error(
                "PyTorch kurulu değil. Önce şu komutu çalıştırın: "
                "python -m pip install -r requirements-core.txt"
            )
        raise

    runtime = create_runtime(arguments.checkpoint, device=arguments.device)
    knowledge_base = None
    if arguments.knowledge_base:
        from hyperai_core.retrieval import LocalKnowledgeBase
        knowledge_base = LocalKnowledgeBase.from_files(arguments.knowledge_base)
        print(f"Yerel bilgi tabanı hazır: {len(knowledge_base.chunks):,} belge parçası.")
    settings = GenerationSettings(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        no_repeat_ngram_size=arguments.no_repeat_ngram_size,
        stop_token_ids=frozenset({ByteTokenizer.EOS_ID}),
    )
    print("HyperAI hazır. Çıkmak için 'exit' yaz.")
    while True:
        prompt = input("Sen: ").strip()
        if prompt.lower() in {"exit", "quit", "çık"}:
            break
        if not prompt:
            continue
        passages = knowledge_base.search(prompt, arguments.retrieval_top_k) if knowledge_base else []
        if arguments.wikipedia:
            try:
                from hyperai_core.web_knowledge import wikipedia_summary
                web_passage = wikipedia_summary(prompt, arguments.web_timeout)
                if web_passage:
                    passages.append(web_passage)
            except (RuntimeError, ValueError) as exc:
                print(f"[Web bilgisi alınamadı: {exc}]")
        _print_sources(passages)
        result = runtime.generate(format_chat_prompt(prompt), settings)
        print(f"HyperAI: {_limit_sentences(result.text, arguments.max_sentences)}")


if __name__ == "__main__":
    main()
