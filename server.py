import os
import sys
import shutil
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import correlate
import librosa
from PIL import Image
import imagehash

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

SR = 22050
HOP_LENGTH = 512
DESIRED_SEGMENT_SEC = 20
MIN_SEGMENT = 10
MAX_SEGMENT = 40
FRAME_FPS = 5
PHASH_SIZE_BITS = 64
AUDIO_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4
EPS = 1e-8

app = FastAPI(title="Video Fingerprint API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

def run_cmd(cmd, capture=False):
  if capture:
    return subprocess.check_output(cmd, shell=True).decode('utf-8')
  else:
    subprocess.check_call(cmd, shell=True)

def extract_audio(video_path, out_path):
  cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", video_path,
    "-ac", "1",
    "-ar", str(SR),
    "-vn", out_path
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, cmd)

def extract_frames(video_path: str, out_dir: str, start: float, duration: float, fps: int = FRAME_FPS):
  os.makedirs(out_dir, exist_ok=True)
  cmd = (
    f'ffmpeg -y -hide_banner -loglevel error -ss {start:.3f} -t {duration:.3f} -i "{video_path}" '
    f'-vf fps={fps} "{os.path.join(out_dir, "frame_%06d.jpg")}"'
  )
  run_cmd(cmd)

def compute_mfcc_mean(wav_path: str):
  y, sr = librosa.load(wav_path, sr=SR, mono=True)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
  mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + EPS)
  vec = mfcc.mean(axis=0)
  return vec, len(vec), y, sr

def choose_segment_by_energy(wav_y: np.ndarray, sr: int, clip_duration: float, desired: int = DESIRED_SEGMENT_SEC) -> Tuple[float, float]:
  desired = int(max(MIN_SEGMENT, min(MAX_SEGMENT, desired)))
  if clip_duration <= desired:
    return 0.0, clip_duration
  hop = int(sr // 2)
  frame_len = int(sr)
  rms = librosa.feature.rms(y=wav_y, frame_length=frame_len, hop_length=hop)[0]
  hop_sec = hop / sr
  win_frames = max(1, int(round(desired / hop_sec)))
  cumsum = np.cumsum(np.concatenate([[0.0], rms]))
  avg_rms = (cumsum[win_frames:] - cumsum[:-win_frames]) / win_frames
  best_idx = int(np.argmax(avg_rms))
  start_seconds = best_idx * hop_sec
  if start_seconds + desired > clip_duration:
    start_seconds = max(0.0, clip_duration - desired)
  return float(start_seconds), float(desired)

def normalized_cross_correlation(v_long: np.ndarray, v_clip: np.ndarray) -> np.ndarray:
  len_long = len(v_long)
  len_clip = len(v_clip)
  if len_long < len_clip:
    return np.array([])
  corr = correlate(v_long, v_clip, mode='valid')
  window = np.ones(len_clip)
  long_energy = np.convolve(v_long * v_long, window, mode='valid')
  clip_energy = np.sum(v_clip * v_clip)
  denom = np.sqrt(long_energy * (clip_energy + EPS)) + EPS
  corr_norm = corr / denom
  return corr_norm

def find_best_audio_match(long_audio_wav: str, clip_audio_wav: str, hop_length=HOP_LENGTH, sr=SR):
  v_long, n_long, y_long, sr_long = compute_mfcc_mean(long_audio_wav)
  v_clip, n_clip, y_clip, sr_clip = compute_mfcc_mean(clip_audio_wav)
  corr_norm = normalized_cross_correlation(v_long, v_clip)
  if corr_norm.size == 0:
    return None
  best_idx = int(np.argmax(corr_norm))
  best_score = float(corr_norm[best_idx])
  frame_rate = sr / hop_length
  start_seconds = best_idx / frame_rate
  duration_seconds = n_clip / frame_rate
  return start_seconds, duration_seconds, best_score

def compute_phash_list(frames_dir: str):
  paths = sorted([p for p in Path(frames_dir).glob("*.jpg")])
  if not paths:
    return []
  phashes = []
  for p in paths:
    try:
      ph = imagehash.phash(Image.open(p))
    except Exception:
      ph = None
    phashes.append((str(p), ph))
  return phashes

def average_hamming_distance(phash_pairs):
  dists = []
  for a, b in phash_pairs:
    if a is None or b is None:
      dists.append(PHASH_SIZE_BITS)
    else:
      d = int(a - b)
      dists.append(d)
  return float(sum(dists) / len(dists))

def confirm_visual_match(long_video: str, clip_segment_start: float, clip_duration: float, clip_frames_dir: str, fps=FRAME_FPS):
  with tempfile.TemporaryDirectory(prefix="vfm_long_frames_") as tmp_ldir:
    extract_frames(long_video, tmp_ldir, clip_segment_start, clip_duration, fps=fps)
    long_ph = compute_phash_list(tmp_ldir)
    clip_ph = compute_phash_list(clip_frames_dir)
    long_hashes = [hp for _, hp in long_ph]
    clip_hashes = [hp for _, hp in clip_ph]
    if not clip_hashes or not long_hashes:
      return None
    L = len(long_hashes)
    C = len(clip_hashes)
    if L < C:
      pairs = list(zip(clip_hashes[:L], long_hashes))
      avg = average_hamming_distance(pairs)
      return avg, L, C
    best_avg = float('inf')
    best_start_idx = 0
    for i in range(0, L - C + 1):
      window = long_hashes[i:i+C]
      pairs = list(zip(window, clip_hashes))
      avg = average_hamming_distance(pairs)
      if avg < best_avg:
        best_avg = avg
        best_start_idx = i
    return best_avg, best_start_idx, C

def extract_clip_segment(full_video: str, out_video: str, start: float, duration: float):
  cmd = f'ffmpeg -y -hide_banner -loglevel error -ss {start:.3f} -t {duration:.3f} -i "{full_video}" -c copy "{out_video}"'
  run_cmd(cmd)

def process_videos(long_video: str, short_clip: str) -> dict:
  tmpdir = tempfile.mkdtemp(prefix='vfm_')
  try:
    long_audio = os.path.join(tmpdir, 'long_audio.wav')
    short_audio = os.path.join(tmpdir, 'short_audio.wav')
    
    extract_audio(long_video, long_audio)
    extract_audio(short_clip, short_audio)
    
    y_short, sr = librosa.load(short_audio, sr=SR, mono=True)
    short_duration = librosa.get_duration(y=y_short, sr=sr)
    
    desired = int(max(MIN_SEGMENT, min(MAX_SEGMENT, DESIRED_SEGMENT_SEC)))
    seg_start, seg_duration = choose_segment_by_energy(y_short, sr, short_duration, desired=desired)
    
    clip_segment_audio = os.path.join(tmpdir, 'clip_segment_audio.wav')
    clip_segment_video = os.path.join(tmpdir, 'clip_segment_video.mp4')
    
    extract_clip_segment(short_clip, clip_segment_video, seg_start, seg_duration)
    extract_audio(clip_segment_video, clip_segment_audio)
    
    audio_match = find_best_audio_match(long_audio, clip_segment_audio)
    if audio_match is None:
      audio_score, audio_start, audio_duration = 0.0, 0.0, 0.0
    else:
      audio_start, audio_duration, audio_score = audio_match
      
    clip_frames_dir = os.path.join(tmpdir, 'clip_frames')
    extract_frames(clip_segment_video, clip_frames_dir, 0.0, seg_duration, fps=FRAME_FPS)
    
    visual_result = confirm_visual_match(long_video, audio_start, seg_duration, clip_frames_dir, fps=FRAME_FPS)
    
    if visual_result is None:
      avg_hamming = None
      visual_percent = 0.0
    else:
      avg_hamming, best_frame_idx, n_frames = visual_result
      visual_percent = max(0.0, 1.0 - (avg_hamming / PHASH_SIZE_BITS))
      
    audio_norm = max(0.0, min(1.0, audio_score)) if audio_match else 0.0
    final_score = AUDIO_WEIGHT * audio_norm + VISUAL_WEIGHT * visual_percent
    
    return {
      "clip_segment": {
        "start_sec": float(seg_start),
        "duration_sec": float(seg_duration)
      },
      "match": {
        "start_sec": float(audio_start),
        "end_sec": float(audio_start + seg_duration)
      },
      "scores": {
        "audio_norm": float(audio_norm),
        "visual_hamming": float(avg_hamming) if avg_hamming is not None else -1.0,
        "visual_percent": float(visual_percent * 100),
        "final_match_percent": float(final_score * 100)
      }
    }
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
  with open("index.html", "r") as f:
    return f.read()

@app.post("/api/match")
async def match_endpoint(long_video: UploadFile = File(...), short_clip: UploadFile = File(...)):
  if shutil.which('ffmpeg') is None:
    raise HTTPException(status_code=500, detail="FFmpeg is not installed on the server.")
    
  with tempfile.TemporaryDirectory() as tmpdir:
    long_path = os.path.join(tmpdir, "long.mp4")
    short_path = os.path.join(tmpdir, "short.mp4")
    
    with open(long_path, "wb") as buffer:
      shutil.copyfileobj(long_video.file, buffer)
    with open(short_path, "wb") as buffer:
      shutil.copyfileobj(short_clip.file, buffer)
      
    try:
      result = process_videos(long_path, short_path)
      return {"status": "success", "data": result}
    except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)