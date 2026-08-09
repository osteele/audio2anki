"""
Sentence selection logic for pipeline: filters transcript segments according to language and content rules.
"""

import logging
import re
from collections.abc import Sequence
from typing import cast

from contextual_langdetect import detect_language, get_majority_language

from .language_types import LanguageCode
from .transcribe import TranscriptionSegment

logger = logging.getLogger(__name__)


def is_empty_or_punctuation(text: str) -> bool:
    return not text or all(c in "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for c in text)


def is_one_word(text: str) -> bool:
    stripped = text.strip()
    # Remove punctuation for the length check
    stripped_no_punct = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~。，？！]', "", stripped)

    # If all characters are CJK Hanzi or Hiragana
    if re.fullmatch(r"[\u4e00-\u9fff\u3040-\u309f]+", stripped_no_punct):
        # For CJK text, only consider it one word if it's 1-2 characters
        return len(stripped_no_punct) <= 2
    # Otherwise, use space-based splitting
    return len(stripped.split()) == 1


def ends_with_comma(text: str) -> bool:
    text = text.strip()
    return text.endswith(",") or text.endswith("，")


def filter_segments(
    segments: Sequence[TranscriptionSegment],
    source_language: LanguageCode | None = None,
) -> list[TranscriptionSegment]:
    """
    Filter segments according to the following rules:
    - Reject one-word segments.
    - Reject segments ending with a comma (standard or CJK).
    - Remove duplicate sentences (keep first occurrence).
    - Reject sentences not in the source language. If not specified, use contextual-langdetect to detect the majority
    language.
    """
    seen: set[str] = set()
    filtered: list[TranscriptionSegment] = []
    # Detect majority language if not specified
    if source_language is None:
        texts = [s.text for s in segments]
        majority_lang = get_majority_language(texts)
        if majority_lang:
            logger.info(f"Detected majority language: {majority_lang}")
            source_language = cast(LanguageCode, str(majority_lang))

    logger.debug(f"Source language: {source_language}")

    def is_duplicate(text: str) -> bool:
        return text in seen

    def has_wrong_language(text: str) -> bool:
        if source_language is None:
            return False
        lang = detect_language(text)
        return lang is not None and lang.language != source_language

    tests = [is_empty_or_punctuation, is_one_word, ends_with_comma, is_duplicate, has_wrong_language]
    for seg in segments:
        text = seg.text.strip()
        logger.debug(f"Processing segment: {text}")

        rejected_reason = next((test.__name__ for test in tests if test(text)), None)
        if rejected_reason is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Rejected segment: {text} ({rejected_reason})")
            continue
        seen.add(text)
        filtered.append(seg)
    return filtered
