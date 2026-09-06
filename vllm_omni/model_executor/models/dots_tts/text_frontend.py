# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts language tags and optional text normalization.

Uses the upstream language libraries and the same WeTextProcessing normalizers
as IndexTTS2/GLM-TTS. Their model-specific punctuation rewriting is deliberately
not applied to dots.tts prompts. Optional dependencies are loaded only on demand.
"""

import re
from functools import lru_cache

_INSTALL_HINT = "Install dots.tts text dependencies with: pip install 'vllm-omni[dots-tts]'"


@lru_cache(maxsize=1)
def _language_detector():
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as exc:
        raise ValueError(_INSTALL_HINT) from exc
    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str) -> str | None:
    language = _language_detector().detect_language_of(text)
    if language is None:
        return None
    code = language.iso_code_639_1 or language.iso_code_639_3
    return code.name.lower()


def resolve_language(language: str | None, text: str) -> str | None:
    if language is None or language.strip().lower() in {"", "none", "unknown"}:
        return None
    language = language.strip()
    if language.lower() in {"auto", "auto_detect"}:
        language = detect_language(text)
        if language is None:
            raise ValueError("Could not detect dots.tts language; supply an explicit language.")
    if language.startswith("口音:"):
        return language
    try:
        from langcodes import Language
    except ImportError as exc:
        raise ValueError(_INSTALL_HINT) from exc
    for resolver in (Language.get, Language.find):
        try:
            resolved = resolver(language).prefer_macrolanguage()
        except (LookupError, ValueError):
            continue
        if resolved.is_valid() and resolved.language and resolved.language != "und":
            code = resolved.language.upper()
            return "口音:粤语" if code == "YUE" else code
    raise ValueError(f"Unsupported dots.tts language: {language!r}")


@lru_cache(maxsize=2)
def _normalizer(language: str):
    try:
        if language == "zh":
            from tn.chinese.normalizer import Normalizer
        else:
            from tn.english.normalizer import Normalizer
    except ImportError as exc:
        raise ValueError(_INSTALL_HINT) from exc
    return Normalizer()


def prepare_text(text: str, ref_text: str | None, *, language: str | None, normalize: bool) -> str:
    if normalize:
        detected = detect_language(text)
        if detected in {"zh", "en"}:
            text = re.sub(r"\s+", " ", _normalizer(detected).normalize(text)).strip()
    resolved_language = resolve_language(language, text)
    body = f"{ref_text}{text}" if ref_text else text
    if resolved_language:
        tag = f"[{resolved_language}]"
        if not body.startswith(tag):
            body = tag + body
    return body
