"""Translation module using OpenAI or DeepL API."""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import deepl
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from rich.progress import Progress, TaskID

from .transcribe import TranscriptionSegment, load_transcript, save_transcript
from .types import LanguageCode
from .utils import create_params_hash

logger = logging.getLogger(__name__)


class TranslationProvider(str, Enum):
    """Supported translation service providers."""

    OPENAI = "openai"
    DEEPL = "deepl"

    @classmethod
    def from_string(cls, value: str) -> "TranslationProvider":
        """Convert a string to an enum value, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OPENAI  # Default to OpenAI


class TranslationParams(TypedDict):
    """Parameters that affect translation output."""

    source_language: LanguageCode | None
    target_language: LanguageCode
    translation_provider: str
    openai_translation_prompt: str
    pinyin_prompt: str
    hiragana_prompt: str
    openai_model: str


# Constants used for translation

# As of 2025-04-22, this appears to be the fastest cheapest model that supports structured output
OPENAI_MODEL: Literal["gpt-4o-mini"] = "gpt-4o-mini"

TRANSLATION_PROMPTS: dict[str, str] = {
    "translation": "You are a translator. Translate the given text to {target_language}.",
    "pinyin": (
        "You are a Chinese language expert. For the given Chinese text:\n"
        "1. Provide Pinyin with tone marks (ā/á/ǎ/à)\n"
        "2. Group syllables into words (no spaces between syllables of the same word)\n"
        "3. Capitalize proper nouns\n"
        "4. Use spaces only between words, not syllables\n"
        "5. Do not include any other text or punctuation\n\n"
        "Examples:\n"
        "Source: 我姓王，你可以叫我小王。\n"
        "Source Pronunciation: wǒ xìngwáng nǐ kěyǐ jiào wǒ Xiǎo Wáng\n\n"
        "Translation: My last name is Wang, you can call me Xiao Wang.\n\n"
        "Source: 他在北京大学学习。\n"
        "Source Pronunciation: tā zài Běijīng Dàxué xuéxí\n"
        "Translation: He studies at Peking University.\n\n"
    ),
    "hiragana": (
        "You are a Japanese language expert. For the given Japanese text:\n"
        "1. Provide hiragana reading\n"
        "2. Keep spaces and punctuation as in the original text\n"
        "3. Do not include any other text or explanations\n\n"
        "Examples:\n"
        "Source: 私は田中です。\n"
        "Source Pronunciation: わたしはたなかです。\n"
        "Translation: I am Tanaka.\n\n"
        "Source: 東京大学で勉強しています。\n"
        "Source Pronunciation: とうきょうだいがくでべんきょうしています。\n"
        "Translation: I am studying at Tokyo University.\n\n"
    ),
}


class TranslationItem(BaseModel):
    start_time: float
    end_time: float
    text: str
    translation: str
    pronunciation: str | None = None


class TranslationResponse(BaseModel):
    items: list[TranslationItem] = Field(..., description="Array of translation results.")


class OpenAIStructuredRefusalError(Exception):
    pass


class OpenAIRateLimitError(Exception):
    pass


class OpenAIInvalidKeyError(Exception):
    pass


def get_translation_hash(
    source_language: LanguageCode | None, target_language: LanguageCode, translation_provider: TranslationProvider
) -> str:
    """
    Generate a hash for the translation function based on its critical parameters.

    This creates a hash of the source language, target language, provider, and the system prompts
    used for translation and reading generation, which will change if any of these parameters
    change, ensuring cached artifacts are invalidated appropriately.

    Args:
        source_language: The source language of the text
        target_language: The target language for translation
        translation_provider: The provider used for translation (OpenAI or DeepL)

    Returns:
        A string hash derived from the parameters
    """

    # Create a dictionary of parameters that affect the output
    params: TranslationParams = {
        "source_language": source_language,
        "target_language": target_language,
        "translation_provider": str(translation_provider),
        "openai_translation_prompt": TRANSLATION_PROMPTS["translation"].format(target_language=target_language),
        "pinyin_prompt": TRANSLATION_PROMPTS["pinyin"],
        "hiragana_prompt": TRANSLATION_PROMPTS["hiragana"],
        "openai_model": OPENAI_MODEL,
    }

    return create_params_hash(params)


def get_pinyin(text: str, model: OpenAIModel) -> str:
    """Get pinyin for Chinese text."""
    agent = Agent(model=model, system_prompt=TRANSLATION_PROMPTS["pinyin"])
    result = agent.run_sync(text)
    return result.output.strip()


def get_hiragana(text: str, model: OpenAIModel) -> str:
    """Get hiragana for Japanese text."""
    agent = Agent(model=model, system_prompt=TRANSLATION_PROMPTS["hiragana"])
    result = agent.run_sync(text)
    return result.output.strip()


def get_reading(text: str, source_language: LanguageCode | None, model: OpenAIModel) -> str | None:
    """Get reading (pinyin or hiragana) based on source language."""
    if not source_language:
        return None
    if source_language.lower() in ["zh", "zh-cn", "zh-tw"]:
        return get_pinyin(text, model)
    elif source_language.lower() in ["ja", "ja-jp"]:
        return get_hiragana(text, model)
    return None


async def translate_with_openai_stream(
    transcript: list[TranscriptionSegment],
    source_language: LanguageCode | None,
    target_language: LanguageCode,
    model: OpenAIModel,
    progress: Progress,
    task_id: TaskID,
) -> TranslationResponse:
    """Translate a transcript using OpenAI API with structured streaming response.

    Shows progress information as segments are translated.
    """
    openai_input: list[dict[str, str | float]] = [
        {"source": s.text, "start_time": s.start, "end_time": s.end} for s in transcript
    ]
    if source_language and source_language.lower() in ["zh", "zh-cn", "zh-tw"]:
        system_prompt = TRANSLATION_PROMPTS["pinyin"]
    elif source_language and source_language.lower() in ["ja", "ja-jp"]:
        system_prompt = TRANSLATION_PROMPTS["hiragana"]
    else:
        system_prompt = TRANSLATION_PROMPTS["translation"].format(target_language=target_language)
    user_prompt = f"Transcript: {json.dumps(openai_input, ensure_ascii=False)}"

    # Create agent with the model and system prompt
    agent = Agent(model=model, system_prompt=system_prompt)

    result: TranslationResponse | None = None
    total_segments = len(transcript)

    # Reset progress to ensure it starts at 0%
    progress.update(task_id, completed=0, total=total_segments)

    async with agent.run_stream(user_prompt, output_type=TranslationResponse) as stream_result:
        async for message, is_final_result in stream_result.stream_structured(debounce_by=0.01):
            try:
                interim_result = await stream_result.validate_structured_output(
                    message, allow_partial=not is_final_result
                )
                if is_final_result:
                    result = interim_result

                completed = len(interim_result.items)
                logger.info(f"Progress update: Completed {completed} of {total_segments} items")

                # We need to force UI refresh for async context
                progress.update(task_id, completed=completed, refresh=True)
                progress.refresh()
            except ValidationError:
                logger.debug("Validation error for message: %s", message)
                if is_final_result:
                    raise

    # Ensure we mark as fully complete at the end
    if result:
        logger.info(f"Final progress update: marking {total_segments} items as complete")
        progress.update(task_id, completed=total_segments, total=total_segments, refresh=True)
        # Force refresh for async context
        progress.refresh()

    if not result:
        raise ValueError("No translation result was received")
    return result


def translate_with_openai_sync(
    transcript: list[TranscriptionSegment],
    source_language: LanguageCode | None,
    target_language: LanguageCode,
    model: OpenAIModel,
    progress: Progress,
    task_id: TaskID,
) -> TranslationResponse:
    """Translate a transcript using OpenAI API with structured response (synchronous version).

    This is a synchronous wrapper around translate_with_openai_stream that handles running the
    async function in the current event loop.
    """

    # Run the async function in the event loop
    return asyncio.run(
        translate_with_openai_stream(
            transcript=transcript,
            source_language=source_language,
            target_language=target_language,
            model=model,
            progress=progress,
            task_id=task_id,
        )
    )


def translate_with_openai(
    transcript: list[TranscriptionSegment],
    source_language: LanguageCode | None,
    target_language: LanguageCode,
    model: OpenAIModel,
) -> TranslationResponse:
    """Translate a transcript using OpenAI API with structured response."""
    openai_input: list[dict[str, str | float]] = [{"source": s.text, "start_time": s.start} for s in transcript]
    if source_language and source_language.lower() in ["zh", "zh-cn", "zh-tw"]:
        system_prompt = TRANSLATION_PROMPTS["pinyin"]
    elif source_language and source_language.lower() in ["ja", "ja-jp"]:
        system_prompt = TRANSLATION_PROMPTS["hiragana"]
    else:
        system_prompt = TRANSLATION_PROMPTS["translation"].format(target_language=target_language)
    user_prompt = f"Transcript: {json.dumps(openai_input, ensure_ascii=False)}"
    # Create agent with the model and system prompt
    agent = Agent(model=model, system_prompt=system_prompt)
    result = agent.run_sync(
        user_prompt,
        output_type=TranslationResponse,
    )
    return result.output


def translate_with_deepl(
    text: str,
    source_language: LanguageCode | None,
    target_language: LanguageCode,
    translator: deepl.Translator,
    model: OpenAIModel | None = None,
) -> tuple[str, str | None]:
    """Translate text using DeepL API.

    Returns:
        Tuple of (translation, reading). Reading is pinyin for Chinese or hiragana for Japanese.
    """
    # Map common language names to DeepL codes
    language_map = {
        "english": "EN-US",
        "chinese": "ZH",
        "japanese": "JA",
        "korean": "KO",
        "french": "FR",
        "german": "DE",
        "spanish": "ES",
        "italian": "IT",
        "portuguese": "PT-BR",
        "russian": "RU",
    }

    target_code = language_map.get(target_language.lower(), target_language.upper())
    result = translator.translate_text(text, target_lang=target_code)

    # Get reading if source is Chinese or Japanese and OpenAI client is available
    reading = None
    if source_language and model:
        reading = get_reading(text, source_language, model)

    # For DeepL, result is a TextResult object or list of TextResult objects
    # For OpenAI, result is already a string
    result_text = getattr(result, "text", None)
    translation = str(result_text if result_text is not None else result)

    return translation, reading


def translate_single_segment(
    segment: TranscriptionSegment,
    source_language: LanguageCode | None,
    target_language: LanguageCode,
    translator: Any,
    use_deepl: bool = False,
    model: OpenAIModel | None = None,
) -> tuple[TranscriptionSegment, TranscriptionSegment | None, bool]:
    """Translate a single segment to target language."""
    try:
        if use_deepl:
            translation, reading = translate_with_deepl(
                segment.text,
                source_language,
                target_language,
                translator,
                model,
            )
        else:
            raise NotImplementedError("OpenAI path should batch segments, not call per-segment translation.")

        translated_segment = TranscriptionSegment(
            start=segment.start,
            end=segment.end,
            text=translation,
        )

        reading_segment: TranscriptionSegment | None = None
        if reading:
            reading_segment = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=reading,
            )

        return translated_segment, reading_segment, True

    except Exception as e:
        print(f"Translation failed: {e!s}")
        return segment, None, False


def normalize_hanzi(text: str) -> str:
    """Normalize hanzi text for comparison by removing spaces and final period.

    Args:
        text: The text to normalize

    Returns:
        Normalized text with spaces removed and final period removed
    """
    # Remove spaces
    text = text.replace(" ", "")
    # Remove final period but keep other punctuation
    if text.endswith("。"):
        text = text[:-1]
    return text


def deduplicate_segments(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    """Deduplicate segments based on normalized hanzi text.

    Args:
        segments: List of segments to deduplicate

    Returns:
        List of segments with duplicates removed, keeping the first occurrence of each normalized text
    """
    seen: set[str] = set()
    result: list[TranscriptionSegment] = []

    for segment in segments:
        normalized = normalize_hanzi(segment.text)
        if normalized not in seen:
            seen.add(normalized)
            result.append(segment)

    return result


def translate_segments(
    input_file: Path,
    output_file: Path,
    target_language: LanguageCode,
    task_id: TaskID,
    progress: Progress,
    source_language: LanguageCode | None = None,
    translation_provider: TranslationProvider = TranslationProvider.OPENAI,
) -> None:
    """Translate transcript and save as JSON file with transcription, translation, and pronunciation."""
    # Check for API keys
    deepl_token = os.environ.get("DEEPL_API_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for translation and readings")

    # Initialize translator and use_deepl flag
    openai_model = OpenAIModel(OPENAI_MODEL)
    translator = openai_model
    use_deepl = False

    # Use the specified translation provider
    if translation_provider == TranslationProvider.DEEPL:
        if deepl_token:
            try:
                logger.info("Attempting to initialize DeepL translator")
                translator = deepl.Translator(deepl_token)
                use_deepl = True
                logger.info("Successfully initialized DeepL translator")
                progress.update(task_id, description="Translating segments using DeepL...")
            except Exception as e:
                print(f"Warning: Failed to initialize DeepL ({e!s}), falling back to OpenAI")
                logger.warning(f"Failed to initialize DeepL: {e}, falling back to OpenAI")
                # Fall back to OpenAI
                translator = openai_model
                use_deepl = False
                progress.update(task_id, description="Translating segments using OpenAI...")
        else:
            print("Warning: DeepL API token not found, falling back to OpenAI")
            logger.warning("DeepL API token not found, falling back to OpenAI")
            # Fall back to OpenAI
            translator = openai_model
            use_deepl = False
            progress.update(task_id, description="Translating segments using OpenAI...")
    else:  # OpenAI
        translator = openai_model
        use_deepl = False
        progress.update(task_id, description="Translating segments using OpenAI...")

    # Load transcript segments
    segments = load_transcript(input_file)
    total_segments = len(segments)
    # Reset progress with proper total and completed=0
    progress.update(task_id, total=total_segments, completed=0)

    # Translate segments
    enriched_segments: list[TranscriptionSegment] = []
    total_success = 0

    if use_deepl:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_seg = {
                executor.submit(
                    translate_single_segment,
                    segment,
                    source_language,
                    target_language,
                    translator,
                    use_deepl,
                    openai_model,  # Always pass OpenAI model for readings
                ): segment
                for segment in segments
            }
            for future in as_completed(future_to_seg):
                translated_seg, reading_segment, success = future.result()
                enriched = TranscriptionSegment(
                    start=translated_seg.start,
                    end=translated_seg.end,
                    text=translated_seg.text,
                    translation=translated_seg.translation,
                    pronunciation=reading_segment.text if reading_segment else None,  # Pronunciation
                )
                enriched_segments.append(enriched)
                if success:
                    total_success += 1
                current_completed = total_success
                logger.info(f"DeepL progress update: Completed {current_completed} of {total_segments} items")
                progress.update(task_id, completed=current_completed, refresh=True)
                progress.refresh()
    else:
        # Use OpenAI: batch all segments at once using the sync wrapper
        response = translate_with_openai_sync(
            segments,
            source_language,
            target_language,
            openai_model,
            progress,
            task_id,
        )
        for seg, item in zip(segments, response.items, strict=False):
            enriched = TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                translation=item.translation,
                pronunciation=item.pronunciation,
            )
            enriched_segments.append(enriched)
            total_success += 1

    # Save all data to a single JSON file
    save_transcript(enriched_segments, output_file)
    # Update progress to 100% complete with the success message
    progress.update(
        task_id,
        completed=total_segments,
        total=total_segments,
        description=f"Translation complete ({total_success}/{total_segments} successful)",
        refresh=True,
    )
