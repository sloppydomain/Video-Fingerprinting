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
AUDIO_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4
EPS = 1e-8
HASH_SIZE = 8                  # side length for each perceptual hash (HASH_SIZE^2 bits)
RANDOM_HASH_SIMILARITY = 0.5  # unrelated perceptual hashes match ~50% of bits by chance
VISUAL_SEARCH_PAD_SEC = 1.5   # slack on each side for the fine pHash alignment search
VISUAL_SCAN_FPS = 1           # frame rate for the independent whole-video visual scan (location only)
MAX_SCAN_FRAMES = 1200        # cap on scan frames so very long videos stay bounded
AGREEMENT_TOL_SEC = 2.0       # audio & visual offsets within this are "in agreement"
DISAGREE_PENALTY = 0.5        # multiply confidence when audio & visual disagree
MATCH_THRESHOLD = 0.5         # final score at/above this is reported as a match

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

def extract_audio(video_path, out_path, start=None, duration=None):
  cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
  if start is not None:
    cmd += ["-ss", f"{start:.3f}"]          # accurate by default when transcoding
  cmd += ["-i", video_path]
  if duration is not None:
    cmd += ["-t", f"{duration:.3f}"]
  cmd += ["-ac", "1", "-ar", str(SR), "-vn", out_path]
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

def compute_mfcc_matrix(wav_path: str):
  y, sr = librosa.load(wav_path, sr=SR, mono=True)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH)
  # Per-coefficient z-score across time: removes the absolute level so matching keys on
  # spectral *shape* (the discriminative part) rather than loudness. We keep the full
  # (n_mfcc, T) matrix instead of averaging the coefficients into one number per frame,
  # which had thrown away almost all of the spectral signature.
  mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + EPS)
  return mfcc, mfcc.shape[1], y, sr

def choose_segment_by_distinctiveness(wav_y: np.ndarray, sr: int, clip_duration: float, desired: int = DESIRED_SEGMENT_SEC) -> Tuple[float, float]:
  # Pick the most *distinctive* window, not the loudest. Onset strength (spectral flux) is
  # high where the audio is changing; a window full of change yields a sharp, unambiguous
  # correlation peak, whereas a loud-but-uniform passage (a sustained tone, a music bed)
  # correlates with many places. Flux is energy-weighted, so near-silent windows score low
  # on their own and are avoided without a separate silence gate.
  desired = int(max(MIN_SEGMENT, min(MAX_SEGMENT, desired)))
  if clip_duration <= desired:
    return 0.0, clip_duration
  onset_env = librosa.onset.onset_strength(y=wav_y, sr=sr, hop_length=HOP_LENGTH)
  frame_rate = sr / HOP_LENGTH
  win_frames = max(1, int(round(desired * frame_rate)))
  if win_frames >= len(onset_env):
    return 0.0, float(min(desired, clip_duration))
  cumsum = np.cumsum(np.concatenate([[0.0], onset_env]))
  win_sum = cumsum[win_frames:] - cumsum[:-win_frames]
  best_idx = int(np.argmax(win_sum))
  start_seconds = best_idx / frame_rate
  if start_seconds + desired > clip_duration:
    start_seconds = max(0.0, clip_duration - desired)
  return float(start_seconds), float(desired)

def normalized_cross_correlation(m_long: np.ndarray, m_clip: np.ndarray) -> np.ndarray:
  # Full-matrix normalized cross-correlation. m_* have shape (n_mfcc, T); we slide the clip
  # across the long video and score every offset using ALL coefficients at once. Result is
  # in [-1, 1] (Cauchy-Schwarz), peaking where the spectral content best aligns.
  n_long = m_long.shape[1]
  n_clip = m_clip.shape[1]
  if n_long < n_clip:
    return np.array([])
  # Numerator: sum of per-coefficient valid cross-correlations.
  num = np.zeros(n_long - n_clip + 1)
  for row_long, row_clip in zip(m_long, m_clip):
    num += correlate(row_long, row_clip, mode='valid')
  # Denominator: sliding L2 norm of the long window times the (constant) clip norm.
  clip_energy = float(np.sum(m_clip * m_clip))
  col_energy = np.sum(m_long * m_long, axis=0)
  window = np.ones(n_clip)
  long_energy = np.convolve(col_energy, window, mode='valid')
  denom = np.sqrt(long_energy * clip_energy) + EPS
  return num / denom

def find_best_audio_match(long_audio_wav: str, clip_audio_wav: str, hop_length=HOP_LENGTH, sr=SR):
  m_long, n_long, y_long, sr_long = compute_mfcc_matrix(long_audio_wav)
  m_clip, n_clip, y_clip, sr_clip = compute_mfcc_matrix(clip_audio_wav)
  corr_norm = normalized_cross_correlation(m_long, m_clip)
  if corr_norm.size == 0:
    return None
  best_idx = int(np.argmax(corr_norm))
  best_score = float(corr_norm[best_idx])
  frame_rate = sr / hop_length
  start_seconds = best_idx / frame_rate
  duration_seconds = n_clip / frame_rate
  return start_seconds, duration_seconds, best_score

def wav_duration(wav_path: str) -> float:
  try:
    return float(librosa.get_duration(path=wav_path))
  except TypeError:                      # older librosa used `filename=`
    return float(librosa.get_duration(filename=wav_path))

def compute_frame_hashes(frames_dir: str):
  # Two complementary perceptual hashes per frame: pHash keys on low-frequency DCT
  # structure, dHash on gradients. They fail on different distortions, so combining them
  # is steadier than either alone.
  paths = sorted(Path(frames_dir).glob("*.jpg"))
  out = []
  for p in paths:
    try:
      img = Image.open(p)
      h = (imagehash.phash(img, hash_size=HASH_SIZE), imagehash.dhash(img, hash_size=HASH_SIZE))
    except Exception:
      h = None
    out.append((str(p), h))
  return out

def frame_distance(a, b) -> float:
  # Normalized [0,1] distance: mean over hash types of (Hamming / bits). Hash-size agnostic.
  if a is None or b is None:
    return 1.0
  dists = [(ha - hb) / ha.hash.size for ha, hb in zip(a, b)]
  return float(sum(dists) / len(dists))

def average_normalized_distance(pairs) -> float:
  if not pairs:
    return 1.0
  return float(sum(frame_distance(a, b) for a, b in pairs) / len(pairs))

def best_sliding_distance(long_hashes, clip_hashes):
  # Slide the clip's frame hashes across the long sequence; return (best_distance, offset_idx).
  L, C = len(long_hashes), len(clip_hashes)
  if L < C:
    return average_normalized_distance(list(zip(clip_hashes[:L], long_hashes))), 0
  best, best_idx = float('inf'), 0
  for i in range(L - C + 1):
    d = average_normalized_distance(list(zip(long_hashes[i:i+C], clip_hashes)))
    if d < best:
      best, best_idx = d, i
  return best, best_idx

def confirm_visual_match(long_video: str, predicted_start: float, clip_duration: float, clip_frames_dir: str, fps=FRAME_FPS, pad=VISUAL_SEARCH_PAD_SEC):
  # Fine, frame-accurate visual score at the audio-predicted location -> the visual quality.
  # Pull a window wider than the clip so the search can slide and lock alignment; snap the
  # pad to a whole number of frames so the long-video sampling phase matches the clip's (a
  # fractional pad shifts every frame off-grid and inflates the distance).
  pad_frames = max(1, round(pad * fps))
  pad = pad_frames / float(fps)
  win_start = max(0.0, predicted_start - pad)
  win_duration = clip_duration + 2 * pad
  with tempfile.TemporaryDirectory(prefix="vfm_long_frames_") as tmp_ldir:
    extract_frames(long_video, tmp_ldir, win_start, win_duration, fps=fps)
    long_hashes = [h for _, h in compute_frame_hashes(tmp_ldir)]
    clip_hashes = [h for _, h in compute_frame_hashes(clip_frames_dir)]
    if not clip_hashes or not long_hashes:
      return None
    return best_sliding_distance(long_hashes, clip_hashes)[0]

def find_best_visual_match(long_video: str, long_duration: float, clip_scan_frames_dir: str, fps=VISUAL_SCAN_FPS):
  # Independent whole-video visual search: where does the clip best match *visually*,
  # ignoring the audio guess? Returns (distance, start_sec) so we can check whether the
  # visual evidence corroborates or contradicts the audio offset, instead of only ever
  # confirming it. Runs at a low fps to stay cheap across the entire source.
  with tempfile.TemporaryDirectory(prefix="vfm_long_scan_") as tmp_ldir:
    extract_frames(long_video, tmp_ldir, 0.0, long_duration, fps=fps)
    long_hashes = [h for _, h in compute_frame_hashes(tmp_ldir)]
    clip_hashes = [h for _, h in compute_frame_hashes(clip_scan_frames_dir)]
    if not clip_hashes or not long_hashes:
      return None
    dist, idx = best_sliding_distance(long_hashes, clip_hashes)
    return dist, idx / float(fps)

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
    seg_start, seg_duration = choose_segment_by_distinctiveness(y_short, sr, short_duration, desired=desired)

    # Pull the segment's audio and frames straight from the clip with accurate seeking.
    # (The old path cut an intermediate file with `-c copy`, which snaps to keyframes and
    # drifts the real start by up to a GOP, misaligning everything downstream.)
    clip_segment_audio = os.path.join(tmpdir, 'clip_segment_audio.wav')
    extract_audio(short_clip, clip_segment_audio, start=seg_start, duration=seg_duration)

    audio_match = find_best_audio_match(long_audio, clip_segment_audio)
    if audio_match is None:
      audio_score, audio_start, audio_duration = 0.0, 0.0, 0.0
    else:
      audio_start, audio_duration, audio_score = audio_match
    audio_norm = max(0.0, min(1.0, audio_score)) if audio_match else 0.0

    # Clip frames at FRAME_FPS for the fine confirmation, and at the (lower) scan fps for
    # the independent whole-video search. Both ends of a comparison must share an fps so the
    # sampling grids line up.
    long_dur = wav_duration(long_audio)
    scan_fps = VISUAL_SCAN_FPS
    if long_dur * scan_fps > MAX_SCAN_FRAMES:
      scan_fps = max(0.5, MAX_SCAN_FRAMES / long_dur)

    clip_frames_dir = os.path.join(tmpdir, 'clip_frames')
    clip_scan_frames_dir = os.path.join(tmpdir, 'clip_scan_frames')
    extract_frames(short_clip, clip_frames_dir, seg_start, seg_duration, fps=FRAME_FPS)
    extract_frames(short_clip, clip_scan_frames_dir, seg_start, seg_duration, fps=scan_fps)

    # Visual score: frame-accurate quality at the audio-predicted spot.
    visual_distance = confirm_visual_match(long_video, audio_start, seg_duration, clip_frames_dir, fps=FRAME_FPS)
    if visual_distance is None:
      visual_distance = 1.0
    # Unrelated hashes collide on ~50% of bits, so "1 - distance" sits at ~0.5 for noise.
    # Rescale so the random baseline maps to 0 and identical frames to 1.
    raw_similarity = 1.0 - visual_distance
    visual_percent = max(0.0, (raw_similarity - RANDOM_HASH_SIMILARITY) / (1.0 - RANDOM_HASH_SIMILARITY))

    # Corroboration: does an independent visual scan land on the same place as the audio?
    scan_result = find_best_visual_match(long_video, long_dur, clip_scan_frames_dir, fps=scan_fps)
    if scan_result is None:
      visual_start = audio_start
      agree = True
    else:
      _, visual_start = scan_result
      agree = abs(visual_start - audio_start) <= AGREEMENT_TOL_SEC

    final_score = AUDIO_WEIGHT * audio_norm + VISUAL_WEIGHT * visual_percent
    if not agree:
      # Audio and visual point at different places -> likely a spurious peak, not a real hit.
      final_score *= DISAGREE_PENALTY
    is_match = final_score >= MATCH_THRESHOLD

    return {
      "clip_segment": {
        "start_sec": float(seg_start),
        "duration_sec": float(seg_duration)
      },
      "match": {
        "start_sec": float(audio_start),
        "end_sec": float(audio_start + seg_duration),
        "visual_start_sec": float(visual_start)
      },
      "decision": {
        "is_match": bool(is_match),
        "verdict": "match" if is_match else "no_match",
        "threshold_percent": float(MATCH_THRESHOLD * 100),
        "audio_visual_agree": bool(agree)
      },
      "scores": {
        "audio_norm": float(audio_norm),
        "visual_distance": float(visual_distance),
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