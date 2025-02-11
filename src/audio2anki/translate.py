"""Translation module using OpenAI or DeepL API."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import deepl
from openai import OpenAI
from rich.progress import Progress, TaskID

from .transcribe import TranscriptionSegment, load_transcript, save_transcript


def get_pinyin(text: str, client: OpenAI) -> str:
    """Get Pinyin for Chinese text using OpenAI."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Chinese language expert. For the given Chinese text:\n"
                    "1. Provide Pinyin with tone marks (ā/á/ǎ/à)\n"
                    "2. Group syllables into words (no spaces between syllables of the same word)\n"
                    "3. Capitalize proper nouns\n"
                    "4. Use spaces only between words, not syllables\n"
                    "5. Do not include any other text or punctuation\n\n"
                    "Examples:\n"
                    "Input: 我姓王，你可以叫我小王。\n"
                    "Output: wǒ xìngwáng nǐ kěyǐ jiào wǒ Xiǎo Wáng\n\n"
                    "Input: 他在北京大学学习。\n"
                    "Output: tā zài Běijīng Dàxué xuéxí"
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    pinyin = response.choices[0].message.content
    if not pinyin:
        raise ValueError("Empty response from OpenAI")
    return pinyin.strip()


def get_hiragana(text: str, client: OpenAI) -> str:
    """Get hiragana for Japanese text using OpenAI."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Japanese language expert. For the given Japanese text:\n"
                    "1. Provide hiragana reading\n"
                    "2. Keep spaces and punctuation as in the original text\n"
                    "3. Do not include any other text or explanations\n\n"
                    "Examples:\n"
                    "Input: 私は田中です。\n"
                    "Output: わたしはたなかです。\n\n"
                    "Input: 東京大学で勉強しています。\n"
                    "Output: とうきょうだいがくでべんきょうしています。"
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    hiragana = response.choices[0].message.content
    if not hiragana:
        raise ValueError("Empty response from OpenAI")
    return hiragana.strip()


def get_reading(text: str, source_language: str, client: OpenAI) -> str | None:
    """Get reading (pinyin or hiragana) based on source language."""
    if source_language.lower() in ["chinese", "zh", "mandarin"]:
        return get_pinyin(text, client)
    elif source_language.lower() in ["japanese", "ja"]:
        return get_hiragana(text, client)
    return None


def translate_with_openai(
    text: str,
    source_language: str | None,
    target_language: str,
    client: OpenAI,
) -> tuple[str, str | None]:
    """Translate text using OpenAI API.

    Returns:
        Tuple of (translation, reading). Reading is pinyin for Chinese or hiragana for Japanese.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a translator. Translate the given text to {target_language}.",
            },
            {"role": "user", "content": f"Translate the following text to {target_language}:\n{text}"},
        ],
    )

    translation = response.choices[0].message.content
    if not translation:
        raise ValueError("Empty response from OpenAI")

    # Get reading if source is Chinese or Japanese
    reading = None
    if source_language:
        reading = get_reading(text, source_language, client)

    return translation.strip(), reading


def translate_with_deepl(
    text: str,
    source_language: str | None,
    target_language: str,
    translator: deepl.Translator,
    openai_client: OpenAI | None = None,
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
    if source_language and openai_client:
        reading = get_reading(text, source_language, openai_client)

    # For DeepL, result is a TextResult object or list of TextResult objects
    # For OpenAI, result is already a string
    result_text = getattr(result, "text", None)
    translation = str(result_text if result_text is not None else result)

    return translation, reading


def translate_single_segment(
    segment: TranscriptionSegment,
    source_language: str | None,
    target_language: str,
    translator: Any,
    use_deepl: bool = False,
    openai_client: OpenAI | None = None,
) -> tuple[TranscriptionSegment, TranscriptionSegment | None, bool]:
    """Translate a single segment to target language.

    Args:
        segment: Audio segment to translate
        source_language: Source language of the text
        target_language: Target language for translation
        translator: Either OpenAI client or DeepL translator
        use_deepl: Whether to use DeepL for translation
        openai_client: OpenAI client for readings when using DeepL

    Returns:
        Tuple of (translated_segment, reading_segment, success flag)
    """
    try:
        if use_deepl:
            translation, reading = translate_with_deepl(
                segment.text,
                source_language,
                target_language,
                translator,
                openai_client,
            )
        else:
            translation, reading = translate_with_openai(
                segment.text,
                source_language,
                target_language,
                translator,
            )

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
        print(f"Translation failed: {str(e)}")
        return segment, None, False


def translate_srt(
    input_file: Path,
    target_language: str,
    task_id: TaskID,
    progress: Progress,
    source_language: str | None = None,
) -> tuple[Path, Path | None]:
    """Translate SRT file to target language.

    Args:
        input_file: Path to input SRT file
        target_language: Target language for translation
        task_id: Task ID for progress tracking
        progress: Progress bar instance
        source_language: Source language of the text (optional)

    Returns:
        Tuple of (translated_file, reading_file). reading_file is None if not applicable.
    """
    # Check for API keys
    deepl_token = os.environ.get("DEEPL_API_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for translation and readings")

    # Initialize OpenAI client for translation and readings
    openai_client = OpenAI(api_key=openai_key)

    # Initialize translator and use_deepl flag
    translator = openai_client  # Default to OpenAI
    use_deepl = False

    # Try DeepL first if available
    if deepl_token:
        try:
            translator = deepl.Translator(deepl_token)
            use_deepl = True
            progress.update(task_id, description="Translating segments using DeepL...")
        except Exception as e:
            print(f"Warning: Failed to initialize DeepL ({str(e)}), falling back to OpenAI")
            deepl_token = None

    # Fall back to OpenAI if DeepL is not available
    if not deepl_token:
        translator = openai_client
        use_deepl = False
        progress.update(task_id, description="Translating segments using OpenAI...")

    # Load segments from SRT file
    segments = load_transcript(input_file)
    total_segments = len(segments)
    progress.update(task_id, total=total_segments)

    # Prepare output files
    translated_file = input_file.with_suffix(f".{target_language}.srt")
    reading_file = None
    if source_language and source_language.lower() in ["chinese", "zh", "mandarin"]:
        reading_file = input_file.with_suffix(".pinyin.srt")
    elif source_language and source_language.lower() in ["japanese", "ja"]:
        reading_file = input_file.with_suffix(".hiragana.srt")

    # Use ThreadPoolExecutor for parallel processing
    translated_segments: list[TranscriptionSegment] = []
    reading_segments: list[TranscriptionSegment] = []
    total_success = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all translation tasks
        future_to_segment = {
            executor.submit(
                translate_single_segment,
                segment,
                source_language,
                target_language,
                translator,
                use_deepl,
                openai_client,  # Always pass OpenAI client for readings
            ): segment
            for segment in segments
        }

        # Process completed translations
        for future in as_completed(future_to_segment):
            translated_segment, reading_segment, success = future.result()
            translated_segments.append(translated_segment)
            if reading_segment:
                reading_segments.append(reading_segment)
            if success:
                total_success += 1
            progress.update(task_id, advance=1)

    # Save translated SRT file
    save_transcript(translated_segments, translated_file)

    # Save reading SRT file if applicable
    if reading_file and reading_segments:
        save_transcript(reading_segments, reading_file)

    progress.update(task_id, description=f"Translation complete ({total_success}/{total_segments} successful)")

    return translated_file, reading_file


# New function to translate a list of segments (used by tests)
def translate_segments(
    segments: list[TranscriptionSegment],
    target_language: str,
    task_id: TaskID,
    progress: Progress,
    source_language: str | None = None,
) -> list[TranscriptionSegment]:
    """Translate a list of transcription segments to the target language.

    This function uses parallel execution to translate each segment using
    translate_single_segment and sets the 'translation' attribute on each segment.

    Returns the list of segments with the added 'translation' attribute.
    """
    # Check for API keys
    deepl_token = os.environ.get("DEEPL_API_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for translation and readings")

    openai_client = OpenAI(api_key=openai_key)
    translator = openai_client  # default
    use_deepl = False

    if deepl_token:
        try:
            import deepl

            translator = deepl.Translator(deepl_token)
            use_deepl = True
        except Exception as e:
            print(f"Warning: Failed to initialize DeepL ({str(e)}), falling back to OpenAI")
            translator = openai_client
            use_deepl = False

    total = len(segments)
    progress.update(task_id, total=total)
    translated_segments: list[TranscriptionSegment] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_seg = {
            executor.submit(
                translate_single_segment, seg, source_language, target_language, translator, use_deepl, openai_client
            ): seg
            for seg in segments
        }
        for future in as_completed(future_to_seg):
            translated_seg, _unused, _unused2 = future.result()
            seg = future_to_seg[future]
            # Set the translation on the segment
            seg.translation = translated_seg.text
            translated_segments.append(seg)
            progress.update(task_id, advance=1)
    return translated_segments
