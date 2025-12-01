
import json
import re
import requests
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from loguru import logger

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

# -------------------------------------------------------------------

class StoryProcessorConfig:
    def __init__(self):
        self.api_url = "https://idsp-dev.teacher-network.in/backend/stories/en/"
        self.base_audio_url = "https://idsp-dev.teacher-network.in/"
        self.request_timeout = 120
        self.trim_seconds = 20
        self.chunk_size = 16384
        self.cache_file = Path("stories_cache.json")
        self.output_csv = Path("stories_analysis.csv")

# -------------------------------------------------------------------

class StoryProcessor:

    def __init__(self):
        self.config = StoryProcessorConfig()

        self.audio_config = AudioNLPConfig()
        self.audio_manager = AudioManager(self.audio_config)

        self.nlp_engine = WhisperNLP(self.audio_config)
        self.nlp_engine.load_models()

        self.writer = ResultWriter(self.config.output_csv)

    def sanitize_filename(self, name):
        s = re.sub(r'[\\/*?:"<>|]', "_", name)
        s = re.sub(r"\s+", "_", s)
        return s.strip("_")[:150]

    def download_audio(self, url, filename):
        out_path = self.audio_config.audio_dir / filename

        if out_path.exists():
            logger.info(f"Audio already exists: {filename}")
            return out_path

        with requests.get(url, stream=True, timeout=self.config.request_timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(out_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename, ncols=80
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=self.config.chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.success(f"Downloaded: {filename}")
        return out_path

    def trim_audio(self, audio_path, seconds):
        trimmed_path = audio_path.with_name(audio_path.stem + "_trimmed" + audio_path.suffix)

        if trimmed_path.exists():
            return trimmed_path

        audio = AudioSegment.from_file(audio_path)
        start_ms = seconds * 1000
        trimmed = audio[start_ms:] if len(audio) > start_ms else audio
        trimmed.export(trimmed_path, format=audio_path.suffix.replace(".", ""))
        return trimmed_path

    def fetch_stories(self):
        if self.config.cache_file.exists():
            with open(self.config.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        resp = requests.get(self.config.api_url, timeout=self.config.request_timeout)
        resp.raise_for_status()
        data = resp.json()

        with open(self.config.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    def process_story(self, story):
        try:
            nid = story.get("nid")
            title = story.get("title") or story.get("story_title") or f"(Untitled {nid})"

            possible_keys = ["audio_story_url", "audio", "audio_url"]
            relative_audio = next((story.get(k) for k in possible_keys if story.get(k)), None)

            if not relative_audio:
                logger.warning(f"No audio for nid={nid}")
                return

            url = self.config.base_audio_url + relative_audio.lstrip("/")
            safe_name = self.sanitize_filename(title)

            ext = url.split("?")[0].split(".")[-1]
            if len(ext) > 5:
                ext = "mp3"

            filename = f"{safe_name}.{ext}"
            audio_path = self.download_audio(url, filename)
            audio_path = self.trim_audio(audio_path, self.config.trim_seconds)

            transcription = self.nlp_engine.transcribe(audio_path)
            text = transcription["text"]

            transcript_path = self.audio_manager.config.transcript_dir / f"{nid}.txt"
            transcript_path.write_text(text, encoding="utf-8")

            analysis = self.nlp_engine.analyze(text)

            analysis_row = {
                "title": title,
                "nid": nid,
                "duration": story.get("duration"),
                "word_count": analysis.get("word_count"),
                "sentence_count": analysis.get("sentence_count"),
                "avg_words_per_sentence": analysis.get("avg_words_per_sentence"),
                "noun_count": analysis["pos_counts"]["nouns"],
                "verb_count": analysis["pos_counts"]["verbs"],
                "adj_count": analysis["pos_counts"]["adjectives"],
            }

            for length in range(3, 10 + 1):
                analysis_row[f"{length}_letter_words"] = analysis["word_length_distribution"].get(length, 0)

            self.writer.write_row(analysis_row)
            logger.success(f"Processed : {title}")

        except Exception:
            logger.exception(f"Failed story nid={story.get('nid')}")


    def run(self):
        stories = self.fetch_stories()
        for s in stories:
            is_community = str(s.get("field_is_it_by_community")).strip() == "1"
            if is_community:
                logger.info(f"Skipping community story: nid={s.get('nid')}")
                continue

            self.process_story(s)
        logger.success("All non-community stories processed.")


# -------------------------------------------------------------------

if __name__ == "__main__":
    StoryProcessor().run()