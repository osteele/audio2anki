"""Audio processing module."""

import hashlib
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from pydub import AudioSegment as PydubSegment  # type: ignore
from pydub.silence import detect_nonsilent  # type: ignore
from rich.progress import Progress, TaskID

from .models import AudioSegment
from .voice_isolation.speechbrain_impl import SpeechBrainVoiceIsolator

logger = logging.getLogger(__name__)

MAX_CLEAN_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
MAX_CHUNK_DURATION = 60 * 1000  # 60 seconds in milliseconds


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    # Return first 8 characters of hash
    return sha256_hash.hexdigest()[:8]


def trim_silence(
    audio: PydubSegment, min_silence_len: int = 100, silence_thresh: int = -50
) -> PydubSegment:
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


def split_large_audio(audio: PydubSegment) -> Iterator[PydubSegment]:
    """Split audio into chunks of maximum 60 seconds each.

    Args:
        audio: Audio segment to split

    Yields:
        Audio segments of at most 60 seconds each
    """
    duration = len(audio)
    for start in range(0, duration, MAX_CHUNK_DURATION):
        end = min(start + MAX_CHUNK_DURATION, duration)
        chunk = audio[start:end]

        # Export to check file size
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            chunk.export(temp_file.name, format="wav")
            size = os.path.getsize(temp_file.name)

            # If chunk is too large, split it in half
            if size > MAX_CLEAN_FILE_SIZE:
                mid = (end - start) // 2
                yield audio[start : start + mid]
                yield audio[start + mid : end]
            else:
                yield chunk


def verify_chunk_limits(chunk: PydubSegment, path: Path) -> None:
    """Verify that a chunk meets the size and duration limits.

    Args:
        chunk: The audio chunk to verify
        path: Path to the exported chunk file

    Raises:
        AudioCleaningError: If the chunk exceeds size or duration limits
    """
    duration = len(chunk)
    size = os.path.getsize(path)

    if duration > MAX_CHUNK_DURATION:
        raise AudioCleaningError(
            f"Chunk duration ({duration / 1000:.1f}s) exceeds limit of 60s. This is a bug - please report it."
        )

    if size > MAX_CLEAN_FILE_SIZE:
        raise AudioCleaningError(
            f"Chunk size ({size / 1024 / 1024:.1f}MB) exceeds limit of 25MB. This is a bug - please report it."
        )


def clean_audio(
    file_path: str | Path,
    progress: Progress,
    task_id: TaskID,
    clean_mode: str = "auto",
) -> Path | None:
    """Clean audio using SpeechBrain source separation."""
    if clean_mode == "skip":
        return None

    try:
        # Initialize SpeechBrain
        progress.update(task_id, description="Loading SpeechBrain model...")
        isolator = SpeechBrainVoiceIsolator()
        
        # Load the audio file
        audio = PydubSegment.from_file(str(file_path))
        duration = len(audio)

        # If file is longer than 60 seconds or larger than 25MB, split it into chunks
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            audio.export(temp_file.name, format="wav")
            size = os.path.getsize(temp_file.name)

        if duration > MAX_CHUNK_DURATION or size > MAX_CLEAN_FILE_SIZE:
            logger.debug(
                f"Audio duration ({duration / 1000:.1f}s) exceeds 60s or size ({size / 1024 / 1024:.1f}MB) exceeds 25MB, "
                "splitting into chunks"
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                chunks = list(split_large_audio(audio))
                chunk_paths = []
                cleaned_audio = PydubSegment.empty()

                # Process each chunk
                for i, chunk in enumerate(chunks, 1):
                    progress.update(
                        task_id,
                        description=f"Processing chunk {i} of {len(chunks)}...",
                    )
                    # Export chunk to temp file
                    chunk_path = temp_dir_path / f"chunk_{i}.wav"
                    chunk.export(chunk_path, format="wav")
                    chunk_paths.append(chunk_path)

                    # Verify chunk size
                    verify_chunk_limits(chunk, chunk_path)

                    # Clean chunk
                    output_path = temp_dir_path / f"cleaned_chunk_{i}.wav"
                    isolator.isolate_voice(chunk_path, output_path)

                    # Load cleaned chunk and append to result
                    cleaned_chunk = PydubSegment.from_wav(str(output_path))
                    cleaned_audio += cleaned_chunk

                # Export combined result
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    cleaned_audio.export(temp_path, format="wav")
                    return temp_path

        else:
            # Process entire file at once
            progress.update(task_id, description="Processing audio file...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                isolator.isolate_voice(Path(file_path), temp_path)
                return temp_path

    except FileNotFoundError as e:
        if 'ffmpeg' in str(e):
            msg = "ffmpeg is not installed. Please install it using your system's package manager."
            logger.error(msg)
            if clean_mode == "force":
                raise AudioCleaningError(msg) from e
            return None
    except Exception as e:
        error_msg = str(e)
        msg = f"Failed to process audio file using SpeechBrain: {error_msg}"
        logger.debug(msg)
        if clean_mode == "force":
            raise AudioCleaningError(msg) from e
        return None


def split_audio(
    input_file: Path,
    segments: list[AudioSegment],
    output_dir: Path,
    task_id: TaskID,
    progress: Progress,
    min_length: float = 0.0,
    max_length: float = float("inf"),
    silence_thresh: int = -40,
    clean_mode: str | None = None,
) -> list[AudioSegment]:
    """Split audio file into segments and trim silence from each segment.

    Args:
        input_file: Path to input audio file
        segments: List of segments to extract
        output_dir: Directory to save extracted segments
        task_id: Task ID for progress tracking
        progress: Progress object for tracking
        min_length: Minimum segment length in seconds
        max_length: Maximum segment length in seconds
        silence_thresh: Silence threshold in dB
        clean_mode: Audio cleaning mode (None, "auto", "speechbrain", etc.)

    Returns:
        List of segments with audio_file paths set
    """
    # Return early if no segments
    if not segments:
        return segments

    # Load audio file
    audio = PydubSegment.from_file(str(input_file))

    # Compute hash of input file
    file_hash = compute_file_hash(input_file)

    # Create media directory
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Process each segment
    for i, segment in enumerate(segments):
        # Skip segments that are too short or too long
        duration = segment.end - segment.start
        if duration < min_length or duration > max_length:
            continue

        # Extract segment audio
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Trim silence
        segment_audio = trim_silence(segment_audio, silence_thresh=silence_thresh)

        # Clean audio if requested
        if clean_mode:
            try:
                # Create temporary file for cleaning
                with (
                    tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in,
                    tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as temp_out,
                ):
                    # Export segment to temp file
                    segment_audio.export(temp_in.name, format="wav")

                    # Clean the audio
                    clean_audio(temp_in.name, progress, task_id, clean_mode=clean_mode)

                    # Load cleaned audio
                    segment_audio = PydubSegment.from_wav(temp_out.name)

                    # Clean up temp files
                    os.unlink(temp_in.name)
                    os.unlink(temp_out.name)
            except AudioCleaningError as e:
                logger.warning(f"Audio cleaning failed for segment {i}: {e}")

        # Export audio segment with hash in filename
        filename = f"audio2anki_{file_hash}_{i + 1:03d}.mp3"
        segment_path = media_dir / filename
        segment_audio.export(segment_path, format="mp3")
        segment.audio_file = filename

        # Update progress
        progress.update(task_id, advance=1)

    return segments


def split_at_silences(
    audio_file: Path,
    min_silence_len: int = 1000,  # 1 second
    silence_thresh: int = -40,  # dB
    max_chunk_size: int = 25 * 1024 * 1024,  # 25MB (OpenAI limit)
    min_chunk_duration: int = 0,  # ms
    max_chunk_duration: int = 60 * 1000,  # 60 seconds
) -> list[tuple[float, float, Path]]:
    """Split audio file at silences into chunks suitable for transcription.

    Args:
        audio_file: Path to input audio file
        min_silence_len: Minimum length of silence in milliseconds
        silence_thresh: Silence threshold in dB
        max_chunk_size: Maximum size of each chunk in bytes
        min_chunk_duration: Minimum duration of each chunk in milliseconds
        max_chunk_duration: Maximum duration of each chunk in milliseconds

    Returns:
        List of tuples containing (start_time, end_time, chunk_path)
    """
    # Load audio file
    audio = PydubSegment.from_file(str(audio_file))

    # Find non-silent sections
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not nonsilent_ranges:
        return []

    # Create temporary directory for chunks
    temp_dir = Path(tempfile.mkdtemp())
    chunks = []

    # Process each non-silent range
    for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
        # Skip chunks that are too short
        duration = end_ms - start_ms
        if duration < min_chunk_duration:
            continue

        # Split long chunks at max_chunk_duration
        current_start = start_ms
        while current_start < end_ms:
            current_end = min(current_start + max_chunk_duration, end_ms)
            chunk_duration = current_end - current_start

            # Extract chunk
            chunk = audio[current_start:current_end]

            # Export chunk
            chunk_path = temp_dir / f"chunk_{i:03d}_{current_start}_{current_end}.wav"
            chunk.export(chunk_path, format="wav")

            # Verify chunk size
            if chunk_path.stat().st_size > max_chunk_size:
                # If chunk is too big, try splitting it in half
                mid_point = current_start + (chunk_duration // 2)
                # Recursively process each half
                first_half = audio[current_start:mid_point]
                second_half = audio[mid_point:current_end]

                # Export halves
                first_path = temp_dir / f"chunk_{i:03d}_{current_start}_{mid_point}.wav"
                second_path = temp_dir / f"chunk_{i:03d}_{mid_point}_{current_end}.wav"
                first_half.export(first_path, format="wav")
                second_half.export(second_path, format="wav")

                # Add both halves if they're within size limit
                if first_path.stat().st_size <= max_chunk_size:
                    chunks.append((current_start / 1000, mid_point / 1000, first_path))
                if second_path.stat().st_size <= max_chunk_size:
                    chunks.append((mid_point / 1000, current_end / 1000, second_path))
            else:
                # Add chunk if it's within size limit
                chunks.append((current_start / 1000, current_end / 1000, chunk_path))

            current_start = current_end

    return chunks


class AudioCleaningError(Exception):
    """Raised when audio cleaning fails."""

    pass


def get_cached_clean_path(file_path: Path) -> Path:
    """Get the path where a cleaned version of the file would be cached.

    Args:
        file_path: Path to original audio file

    Returns:
        Path where cleaned version would be cached
    """
    file_hash = compute_file_hash(file_path)
    return file_path.parent / f"{file_path.stem}.cleaned-{file_hash}{file_path.suffix}"
