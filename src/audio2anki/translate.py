"""Translation module using OpenAI or DeepL API."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import deepl
from openai import OpenAI
from rich.progress import Progress, TaskID

from .models import AudioSegment


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


def translate_with_openai(
    text: str,
    source_language: str | None,
    target_language: str,
    client: OpenAI,
) -> tuple[str, str | None]:
    """Translate text using OpenAI API.

    Returns:
        Tuple of (translation, pinyin). Pinyin is only provided for Chinese source text.
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

    # Get Pinyin if source is Chinese
    pinyin = None
    if source_language and source_language.lower() in ["chinese", "zh", "mandarin"]:
        pinyin = get_pinyin(text, client)

    return translation.strip(), pinyin


def translate_with_deepl(
    text: str,
    source_language: str | None,
    target_language: str,
    translator: deepl.Translator,
    openai_client: OpenAI | None = None,
) -> tuple[str, str | None]:
    """Translate text using DeepL API.

    Returns:
        Tuple of (translation, pinyin). Pinyin is only provided for Chinese source text.
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

    # Get Pinyin if source is Chinese and OpenAI client is available
    pinyin = None
    if source_language and source_language.lower() in ["chinese", "zh", "mandarin"] and openai_client:
        pinyin = get_pinyin(text, openai_client)

    # For DeepL, result is a TextResult object or list of TextResult objects
    # For OpenAI, result is already a string
    result_text = getattr(result, "text", None)
    translation = str(result_text if result_text is not None else result)

    return translation, pinyin


def translate_single_segment(
    segment: AudioSegment,
    source_language: str | None,
    target_language: str,
    translator: Any,
    use_deepl: bool = False,
    openai_client: OpenAI | None = None,
) -> tuple[AudioSegment, bool]:
    """Translate a single segment to target language.

    Args:
        segment: Audio segment to translate
        source_language: Source language of the text
        target_language: Target language for translation
        translator: Either OpenAI client or DeepL translator
        use_deepl: Whether to use DeepL for translation
        openai_client: OpenAI client for Pinyin when using DeepL

    Returns:
        Tuple of (segment, success flag)
    """
    if segment.translation is not None:
        return segment, True

    try:
        if use_deepl:
            translation, pinyin = translate_with_deepl(
                segment.text,
                source_language,
                target_language,
                translator,
                openai_client,
            )
        else:
            translation, pinyin = translate_with_openai(
                segment.text,
                source_language,
                target_language,
                translator,
            )

        segment.translation = translation
        if pinyin:
            segment.pronunciation = pinyin
        return segment, True

    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return segment, False


def translate_segments(
    segments: list[AudioSegment],
    target_language: str,
    task_id: TaskID,
    progress: Progress,
    source_language: str | None = None,
) -> list[AudioSegment]:
    """Translate segments to target language using parallel processing."""
    # Check for API keys
    deepl_token = os.environ.get("DEEPL_API_TOKEN")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for translation and Pinyin")

    # Initialize OpenAI client for translation and Pinyin
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

    # Use ThreadPoolExecutor for parallel processing
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
                openai_client,  # Always pass OpenAI client for Pinyin
            ): segment
            for segment in segments
        }

        # Process completed translations
        translated_segments: list[AudioSegment] = []
        total_success = 0
        for future in as_completed(future_to_segment):
            segment, success = future.result()
            translated_segments.append(segment)
            if success:
                total_success += 1
            progress.update(task_id, advance=1)

        progress.update(task_id, description=f"Translation complete ({total_success}/{len(segments)} successful)")

    return translated_segments
