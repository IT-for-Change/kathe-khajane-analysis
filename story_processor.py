import sys
import re
import csv
from pathlib import Path
from tqdm import tqdm
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_to_insights import (
    AudioNLPConfig,
    AudioManager,
    WhisperNLP,
    ResultWriter
)

logger.add(
    "story_processor.log",
    rotation="10 MB",
    retention="10 days",
    backtrace=True,
    diagnose=True
)


class StoryProcessorConfig:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.local_audio_dir = self.base_dir / "audio_inputs"
        self.output_csv = self.base_dir / "stories_analysis.csv"


class StoryProcessor:

    def __init__(self):
        self.config = StoryProcessorConfig()

        self.audio_config = AudioNLPConfig()
        self.audio_manager = AudioManager(self.audio_config)

        self.nlp_engine = WhisperNLP(self.audio_config)
        self.nlp_engine.load_models()

        self.writer = ResultWriter(self.config.output_csv)

        # Load already processed audios from CSV
        self.processed_audios = self._load_processed_audios()

    def _load_processed_audios(self) -> set:
        """Read CSV and return set of audio_file names already processed"""
        processed = set()

        if self.config.output_csv.exists():
            with self.config.output_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed.add(row["audio_file"])

            logger.info(f"Loaded {len(processed)} already processed audios")

        return processed

    def list_audio_files(self):
        return sorted([
            p for p in self.config.local_audio_dir.iterdir()
            if p.suffix.lower() in {".mp3", ".wav", ".m4a"}
        ])

    def process_audio(self, audio_path: Path):
        try:

            if audio_path.name in self.processed_audios:
                logger.info(f"Skipping (already analyzed): {audio_path.name}")
                return

            logger.info(f"Processing audio: {audio_path.name}")

            transcript_path = (
                self.audio_manager.config.transcript_dir
                / f"{audio_path.stem}.txt"
            )

            if transcript_path.exists():
                logger.info(f"Using existing transcript: {transcript_path.name}")
                text = transcript_path.read_text(encoding="utf-8")
            else:
                transcription = self.nlp_engine.transcribe(audio_path)
                text = transcription["text"]
                transcript_path.write_text(text, encoding="utf-8")

            analysis = self.nlp_engine.analyze(text)

            row = {
                "audio_file": audio_path.name,
                "word_count": analysis["word_count"],
                "sentence_count": analysis["sentence_count"],
                "avg_words_per_sentence": analysis["avg_words_per_sentence"],
                "noun_count": analysis["pos_counts"]["nouns"],
                "verb_count": analysis["pos_counts"]["verbs"],
                "adj_count": analysis["pos_counts"]["adjectives"],
            }

            for length in range(3, 11):
                row[f"{length}_letter_words"] = (
                    analysis["word_length_distribution"].get(length, 0)
                )

            self.writer.write_row(row)
            self.processed_audios.add(audio_path.name)

            logger.success(f"Done: {audio_path.name}")

        except Exception:
            logger.exception(f"Failed audio: {audio_path.name}")

    def run(self):
        audio_files = self.list_audio_files()

        if not audio_files:
            logger.warning("No audio files found")
            return

        for audio in tqdm(audio_files, desc="Processing audios"):
            self.process_audio(audio)

        logger.success("All local audios processed")


if __name__ == "__main__":
    StoryProcessor().run()
