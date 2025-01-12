"""Translation module using OpenAI."""

import os
from typing import Any

import openai
from rich.progress import Progress

from .cli import AudioSegment


def translate_segments(
    segments: list[AudioSegment],
    task_id: int,
    progress: Progress,
) -> list[AudioSegment]:
    """Translate segments using OpenAI."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it to use translation features."
        )
    
    client = openai.OpenAI()
    
    for i, segment in enumerate(segments):
        # Create prompt for translation and pronunciation
        prompt = f"""Translate this text to English and provide pronunciation. Format as JSON:
Text: {segment.text}

Return format:
{{
    "translation": "English translation",
    "pronunciation": "pronunciation (pinyin for Chinese, romaji for Japanese, etc.)"
}}
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful language translation assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from OpenAI")
            
            # Parse response
            data = eval(result)  # Safe since we specified json_object format
            segment.translation = data["translation"]
            segment.pronunciation = data["pronunciation"]
            
            progress.update(task_id, advance=1)
            
        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    return segments
