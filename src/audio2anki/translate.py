"""Translation module using OpenAI or DeepL API."""

import os
from concurrent.futures import ThreadPoolExecutor

from deepl import Translator as DeepLTranslator
from deepl.api_data import TextResult
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
    # Detect if text contains Chinese characters
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a translator. Translate the following text to {target_language}. "
                    "Return only the translation, no explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    translation = response.choices[0].message.content
    if not translation:
        raise ValueError("Empty response from OpenAI")

    # Get Pinyin if source text contains Chinese characters
    pinyin = get_pinyin(text, client) if has_chinese else None

    return translation.strip(), pinyin


def translate_with_deepl(
    text: str,
    source_language: str | None,
    target_language: str,
    translator: DeepLTranslator,
    openai_client: OpenAI | None = None,
) -> tuple[str, str | None]:
    """Translate text using DeepL API.

    Returns:
        Tuple of (translation, pinyin). Pinyin is only provided for Chinese source text.
    """
    # Detect if text contains Chinese characters
    has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)

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
    translation = process_text_result(result)

    # Get Pinyin if source text contains Chinese characters and OpenAI client is available
    pinyin = get_pinyin(text, openai_client) if has_chinese and openai_client else None

    return translation, pinyin


def translate_single_segment(
    segment: AudioSegment,
    source_language: str | None,
    target_language: str,
    translator: DeepLTranslator | OpenAI,
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
        if use_deepl and isinstance(translator, DeepLTranslator):
            translation, pinyin = translate_with_deepl(
                segment.text,
                source_language,
                target_language,
                translator,
                openai_client,
            )
        elif isinstance(translator, OpenAI):
            translation, pinyin = translate_with_openai(
                segment.text,
                source_language,
                target_language,
                translator,
            )
        else:
            raise ValueError("Unsupported translator type")

        segment.translation = translation
        if pinyin:
            segment.pronunciation = pinyin
        return segment, True

    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return segment, False


def translate_segments(
    segments: list[AudioSegment],
    task_id: TaskID,
    progress: Progress,
    target_language: str = "en",
    use_deepl: bool = False,
) -> list[AudioSegment]:
    """Translate segments using either OpenAI or DeepL."""
    if not segments:
        return segments

    # Initialize translation client
    if use_deepl:
        deepl_key = os.environ.get("DEEPL_API_KEY")
        if not deepl_key:
            raise ValueError("DEEPL_API_KEY environment variable not set")
        translator = DeepLTranslator(auth_key=deepl_key)
    else:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        translator = OpenAI()

    # Process segments in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for segment in segments:
            if segment.text:
                if use_deepl:
                    futures.append(
                        executor.submit(
                            lambda t: translate_with_deepl(t, None, target_language, translator),
                            segment.text,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            lambda t: translate_with_openai(t, None, target_language, translator),
                            segment.text,
                        )
                    )

        # Process results
        translated_segments = []
        for segment, future in zip(segments, futures, strict=True):
            try:
                if future:
                    translation, pinyin = future.result()
                    segment.translation = translation
                    if pinyin:
                        segment.pronunciation = pinyin
                translated_segments.append(segment)
                progress.update(task_id, advance=1)
            except Exception as e:
                raise RuntimeError(f"Error translating segment: {str(e)}") from e
                segment.translation = None
                translated_segments.append(segment)
                progress.update(task_id, advance=1)

    return translated_segments


def detect_language(text: str) -> str:
    """Detect language of text using OpenAI."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a language detector. Return only the ISO 639-1 language code for the given text.",
            },
            {"role": "user", "content": text},
        ],
    )
    result = response.choices[0].message.content
    return result.strip() if result else "en"


def process_text_result(result: TextResult | list[TextResult]) -> str:
    """Process text result from DeepL API."""
    if isinstance(result, list):
        # Handle empty list case
        if not result:
            return ""
        # Join texts from all results
        return " ".join(r.text for r in result)
    # Single result case
    return result.text


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    translator: DeepLTranslator | OpenAI,
    task_id: TaskID,
    progress: Progress,
) -> str:
    """Translate text using the provided translator."""
    try:
        if isinstance(translator, DeepLTranslator):
            result = translator.translate_text(text=text, target_lang=target_lang, source_lang=source_lang)
            return process_text_result(result)
        elif isinstance(translator, OpenAI):
            response = translator.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate to {target_lang}. Return only translation.",
                    },
                    {"role": "user", "content": text},
                ],
            )
            return response.choices[0].message.content or text
        else:
            raise ValueError(f"Unsupported translator type: {type(translator)}")
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}") from e
