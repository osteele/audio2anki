import logging
import os

import click

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Configuration defaults
DEFAULT_CONFIG = {
    "clean_files": True,
    "use_cache": True,
    "cache_expiry_days": 7,
    "voice_isolation_provider": "eleven_labs",
    "transcription_provider": "openai_whisper",
}


def load_config():
    """Load configuration from $HOME/.config/audio2anki/config.toml and return a config dictionary."""
    config_path = os.path.join(os.environ.get("HOME", "."), ".config", "audio2anki", "config.toml")
    config = DEFAULT_CONFIG.copy()
    try:
        with open(config_path, "r") as f:
            # For now, we use a simple parsing: ignore comments and simple key = value pairs
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # A simple conversion: if value is a digit, convert to int, else leave as string or bool
                        if value.lower() in ["true", "false"]:
                            config[key] = value.lower() == "true"
                        elif value.isdigit():
                            config[key] = int(value)
                        else:
                            config[key] = value
    except FileNotFoundError:
        logging.info(f"Config file not found at {config_path}. Using default configuration.")
    return config


def init_cache():
    """Ensure the cache directory exists and return its path."""
    cache_dir = os.path.join(os.environ.get("HOME", "."), ".cache", "audio2anki")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Cache directory created at {cache_dir}")
    else:
        logging.info(f"Cache directory exists at {cache_dir}")
    return cache_dir


def transcode(input_file: str) -> str:
    """Transcode an audio/video file to an audio file suitable for processing. Returns path to audio file."""
    logging.info(f"Transcoding {input_file}...")
    # Placeholder: In a real implementation, call ffmpeg to extract audio.
    audio_file = input_file + ".audio.mp3"
    logging.info(f"Transcoded to {audio_file}")
    return audio_file


def voice_isolation(audio_file: str) -> str:
    """Perform voice isolation using the selected provider. Returns path to cleaned audio file."""
    logging.info(f"Performing voice isolation on {audio_file} using Eleven Labs API...")
    # Placeholder: In a real implementation, call the Eleven Labs API.
    cleaned_audio = audio_file.replace(".audio.mp3", ".cleaned.mp3")
    logging.info(f"Voice isolation complete. Output: {cleaned_audio}")
    return cleaned_audio


def transcribe(audio_file: str) -> str:
    """Transcribe the audio file using OpenAI Whisper and produce an SRT file. Returns path to SRT file."""
    logging.info(f"Transcribing {audio_file} using OpenAI Whisper...")
    # Placeholder: In a real implementation, call OpenAI Whisper.
    srt_file = audio_file.replace(".cleaned.mp3", ".srt")
    logging.info(f"Transcription complete. SRT file generated: {srt_file}")
    return srt_file


def sentence_selection(transcript: str) -> str:
    """Process the transcript to perform sentence selection. Returns path to the processed transcript."""
    logging.info(f"Performing sentence selection on transcript {transcript}...")
    # Placeholder: In a real implementation, apply sentence splitting/filtering.
    selected_transcript = transcript.replace(".srt", ".selected.txt")
    logging.info(f"Sentence selection complete. Output: {selected_transcript}")
    return selected_transcript


def generate_deck(processed_data: str) -> None:
    """Generate an Anki flashcard deck from the processed data."""
    logging.info(f"Generating card deck from {processed_data}...")
    # Placeholder: In a real implementation, create a directory structure with deck.txt, media folder, README, etc.
    logging.info("Card deck generation complete.")


def run_pipeline(input_file: str):
    """Run the entire pipeline for the given input file."""
    logging.info("Starting pipeline...")

    # Transcode input file
    audio_file = transcode(input_file)

    # Voice Isolation
    cleaned_audio = voice_isolation(audio_file)

    # Transcription
    srt_file = transcribe(cleaned_audio)

    # Optional Sentence Selection
    processed_transcript = sentence_selection(srt_file)

    # Card Deck Generation
    generate_deck(processed_transcript)

    logging.info("Pipeline execution complete.")


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug output")
def main(input_file, debug):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled.")

    config = load_config()
    cache_dir = init_cache()

    logging.info(f"Loaded configuration: {config}")
    logging.info(f"Using cache directory: {cache_dir}")

    run_pipeline(input_file)


if __name__ == "__main__":
    main()
