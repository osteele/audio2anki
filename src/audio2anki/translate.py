"""Translation module using OpenAI or DeepL API."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.progress import Progress, TaskID
from openai import OpenAI
import deepl

from .models import AudioSegment


def translate_with_openai(text: str, target_language: str, client: OpenAI) -> str:
    """Translate text using OpenAI API."""
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
    return translation.strip()


def translate_with_deepl(text: str, target_language: str, translator: deepl.Translator) -> str:
    """Translate text using DeepL API."""
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
    return result.text


def translate_single_segment(
    segment: AudioSegment,
    target_language: str,
    translator: Any,
    use_deepl: bool = False,
) -> tuple[AudioSegment, bool]:
    """Translate a single segment to target language.
    
    Args:
        segment: Audio segment to translate
        target_language: Target language for translation
        translator: Either OpenAI client or DeepL translator
        use_deepl: Whether to use DeepL for translation
    
    Returns:
        Tuple of (segment, success flag)
    """
    if segment.translation is not None:
        return segment, True

    try:
        if use_deepl:
            translation = translate_with_deepl(segment.text, target_language, translator)
        else:
            translation = translate_with_openai(segment.text, target_language, translator)

        segment.translation = translation
        return segment, True

    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return segment, False


def translate_segments(
    segments: list[AudioSegment],
    target_language: str,
    task_id: TaskID,
    progress: Progress,
) -> list[AudioSegment]:
    """Translate segments to target language using parallel processing."""
    # Check for DeepL API token first
    deepl_token = os.environ.get("DEEPL_API_TOKEN")
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
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Neither DEEPL_API_TOKEN nor OPENAI_API_KEY environment variable is set")
        translator = OpenAI(api_key=api_key)
        use_deepl = False
        progress.update(task_id, description="Translating segments using OpenAI...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all translation tasks
        future_to_segment = {
            executor.submit(translate_single_segment, segment, target_language, translator, use_deepl): segment
            for segment in segments
        }

        # Process completed translations
        translated_segments = []
        for future in as_completed(future_to_segment):
            segment, success = future.result()
            translated_segments.append(segment)
            progress.update(task_id, advance=1)

        progress.update(task_id, description="Translation complete")

    return translated_segments
