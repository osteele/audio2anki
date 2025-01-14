"""Translation module using OpenAI API."""

import os
from rich.progress import Progress, TaskID
from openai import OpenAI
from .models import AudioSegment


def translate_segments(
    segments: list[AudioSegment],
    target_language: str,
    task_id: TaskID,
    progress: Progress,
) -> list[AudioSegment]:
    """Translate segments to target language."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize OpenAI client
    client = OpenAI()

    # Process each segment
    for segment in segments:
        # Skip if already translated
        if segment.translation is not None:
            progress.update(task_id, advance=1)
            continue

        try:
            # Create prompt
            prompt = f"Translate the following text to {target_language}:\n{segment.text}"

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

            segment.translation = translation

            # Update progress
            progress.update(task_id, advance=1)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Translation failed: {str(e)}")

    return segments
