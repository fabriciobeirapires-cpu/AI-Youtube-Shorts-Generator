"""SamurAI no arquivo local — usa Groq Whisper (API) em vez de faster-whisper local."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from groq import Groq

from shorts_generator.highlights import get_highlights
from shorts_generator.local.clipper import crop_highlights_local
from shorts_generator.local.llm import call_openai_llm

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def transcribe_groq(video_path: str, language: str = "pt") -> dict:
    print(f"[runner] extraindo audio...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", audio_path,
        ], check=True, capture_output=True)

        print(f"[runner] transcrevendo via Groq Whisper...", flush=True)
        with open(audio_path, "rb") as f:
            r = groq_client.audio.transcriptions.create(
                file=(Path(audio_path).name, f, "audio/mpeg"),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language=language,
            )
    finally:
        os.unlink(audio_path)

    raw = r.segments if hasattr(r, "segments") else r["segments"]

    def get(o, k):
        return getattr(o, k) if hasattr(o, k) else o[k]

    segments = [{"start": float(get(s, "start")), "end": float(get(s, "end")), "text": get(s, "text").strip()} for s in raw]
    duration = segments[-1]["end"] if segments else 0
    print(f"[runner] {len(segments)} segmentos | {duration:.0f}s", flush=True)
    return {"duration": duration, "segments": segments}


def run(video_path: str, num_clips: int, aspect_ratio: str, language: str):
    source_path = str(Path(video_path).resolve())
    print(f"[runner] arquivo: {source_path}", flush=True)

    transcript = transcribe_groq(source_path, language)

    print("[runner] ranqueando highlights via Groq LLM...", flush=True)
    result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_openai_llm)
    all_h = result.get("highlights", [])
    print(f"[runner] {len(all_h)} candidatos", flush=True)

    top = sorted(all_h, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[runner] cortando top {len(top)}...", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio)

    return {"source_video_url": source_path, "transcript": transcript, "highlights": all_h, "shorts": shorts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--num-clips", type=int, default=5)
    p.add_argument("--aspect-ratio", default="9:16")
    p.add_argument("--language", default="pt")
    p.add_argument("--output-json", default="result.json")
    args = p.parse_args()

    try:
        r = run(args.video, args.num_clips, args.aspect_ratio, args.language)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 72)
    for i, s in enumerate(r["shorts"], 1):
        print(f"\n#{i} score={s.get('score')} {s.get('start_time'):.1f}s -> {s.get('end_time'):.1f}s")
        print(f"   titulo: {s.get('title')}")
        print(f"   hook:   {s.get('hook_sentence')}")
        print(f"   clip:   {s.get('clip_url') or 'FAILED: ' + str(s.get('error'))}")

    slim = {k: v for k, v in r.items() if k != "transcript"}
    slim["transcript_duration"] = r["transcript"].get("duration", 0)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
