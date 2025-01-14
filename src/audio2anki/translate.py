"""Translation module using OpenAI API."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, TaskID
from openai import OpenAI
from .models import AudioSegment


def translate_single_segment(
    segment: AudioSegment,
    target_language: str,
    api_key: str,
) -> tuple[AudioSegment, bool]:
    """Translate a single segment to target language.
    
    Returns:
        Tuple of (segment, success flag)
    """
    if segment.translation is not None:
        return segment, True

    try:
        # Create prompt
        prompt = f"Translate the following text to {target_language}:\n{segment.text}"

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the given text to {target_language}.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Extract translation
        translation = response.choices[0].message.content
        if not translation:
            raise ValueError("Empty response from OpenAI")

        # Update segment
        segment.translation = translation.strip()
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
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all translation tasks
        future_to_segment = {
            executor.submit(translate_single_segment, segment, target_language, api_key): segment
            for segment in segments
        }

        # Process completed translations
        translated_segments = []
        for future in as_completed(future_to_segment):
            segment, success = future.result()
            translated_segments.append(segment)
            progress.update(task_id, advance=1)

    return translated_segments
