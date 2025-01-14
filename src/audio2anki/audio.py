"""Audio processing module."""

from pathlib import Path

from pydub import AudioSegment as PydubSegment  # type: ignore
from pydub.silence import detect_nonsilent  # type: ignore
from rich.progress import Progress, TaskID

from .models import AudioSegment


def trim_silence(audio: PydubSegment, min_silence_len: int = 100, silence_thresh: int = -50) -> PydubSegment:
    """Trim silence from the beginning and end of an audio segment.

    Args:
        audio: Audio segment to trim
        min_silence_len: Minimum length of silence in milliseconds
        silence_thresh: Silence threshold in dB

    Returns:
        Trimmed audio segment
    """
    # Find non-silent sections
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not nonsilent_ranges:
        return audio

    # Get start and end of non-silent audio
    start_trim = nonsilent_ranges[0][0]
    end_trim = nonsilent_ranges[-1][1]

    return audio[start_trim:end_trim]


def split_audio(
    input_file: Path,
    segments: list[AudioSegment],
    output_dir: Path,
    task_id: TaskID,
    progress: Progress,
    silence_thresh: int = -40,
) -> list[AudioSegment]:
    """Split audio file into segments and trim silence from each segment.

    Args:
        input_file: Path to input audio file
        segments: List of segments to extract
        output_dir: Directory to save audio segments
        task_id: Progress bar task ID
        progress: Progress bar instance
        silence_thresh: Silence threshold in dB. Higher (less negative) values mean more
            aggressive silence detection. Default is -40dB.

    Returns:
        List of segments with updated paths to audio files
    """
    # Load audio file
    audio = PydubSegment.from_file(str(input_file))

    # Create media directory
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Process each segment
    for i, segment in enumerate(segments):
        # Extract segment audio
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Trim silence
        segment_audio = trim_silence(segment_audio, silence_thresh=silence_thresh)

        # Export audio segment
        segment_path = media_dir / f"segment_{i+1:03d}.mp3"
        segment_audio.export(segment_path, format="mp3")
        segment.audio_path = segment_path

        # Update progress
        progress.update(task_id, advance=1)

    return segments
