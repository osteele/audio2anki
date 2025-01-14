"""Audio processing module."""

from pathlib import Path

from pydub import AudioSegment as PydubSegment  # type: ignore
from rich.progress import Progress, TaskID

from .models import AudioSegment


def split_audio(
    input_file: Path,
    segments: list[AudioSegment],
    output_dir: Path,
    task_id: TaskID,
    progress: Progress,
) -> list[AudioSegment]:
    """Split audio file into segments."""
    # Load audio file
    audio = PydubSegment.from_file(str(input_file))
    
    # Create media directory
    media_dir = output_dir / "media"
    media_dir.mkdir(exist_ok=True)
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Convert timestamps to milliseconds
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        
        # Extract segment
        audio_segment = audio[start_ms:end_ms]
        
        # Generate filename
        filename = f"audio_{i+1:04d}.mp3"
        output_path = media_dir / filename
        
        # Export
        audio_segment.export(
            output_path,
            format="mp3",
            parameters=["-q:a", "0"],  # Use highest quality
        )
        
        # Update segment
        segment.audio_file = filename
        
        # Update progress
        progress.update(task_id, advance=1)
    
    return segments
