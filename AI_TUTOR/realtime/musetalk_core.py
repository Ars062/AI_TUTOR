"""Minimal MuseTalk 1.5 inference core for a static portrait.

Implements only what the realtime avatar needs (no mmpose / face-parse /
librosa / mmlab): whisper audio features -> UNet inpainting -> SD-VAE decode
-> feathered paste of the lower face back onto the portrait.

Weights layout (models/musetalk):
  musetalkV15/unet.pth + musetalkV15/musetalk.json
  sd-vae-ft-mse/{config.json,diffusion_pytorch_model.bin}
  whisper/{config.json,pytorch_model.bin,preprocessor_config.json}
"""
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoFeatureExtractor, WhisperModel

AUDIO_SR = 16000
AUDIO_FPS = 50
PAD_LEFT = 2
PAD_RIGHT = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        b, seq_len, d_model = x.size()
        return x + self.pe[:, :seq_len, :].to(device=x.device, dtype=x.dtype)


def resample_to_16k(audio_f32: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == AUDIO_SR:
        return audio_f32
    n = audio_f32.size
    if n < 2:
        return audio_f32
    target = max(1, int(round(n * AUDIO_SR / sample_rate)))
    x_old = np.linspace(0.0, n - 1.0, n)
    x_new = np.linspace(0.0, n - 1.0, target)
    return np.interp(x_new, x_old, audio_f32).astype(np.float32)


class MuseTalkCore:
    def __init__(self, weights_dir: str, device: str = "cuda", batch_size: int = 8):
        t0 = time.time()
        self.weights_dir = weights_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.batch_size = batch_size

        vae_dir = os.path.join(weights_dir, "sd-vae-ft-mse")
        self.vae = AutoencoderKL.from_pretrained(vae_dir).to(self.device, self.dtype)
        self.vae.eval()
        self.scaling = float(self.vae.config.scaling_factor)

        with open(os.path.join(weights_dir, "musetalkV15", "musetalk.json")) as f:
            unet_cfg = json.load(f)
        unet_cfg = {k: v for k, v in unet_cfg.items() if not k.startswith("_")}
        self.unet = UNet2DConditionModel(**unet_cfg)
        sd = torch.load(
            os.path.join(weights_dir, "musetalkV15", "unet.pth"),
            map_location="cpu",
            weights_only=True,
        )
        self.unet.load_state_dict(sd)
        self.unet = self.unet.to(self.device, self.dtype).eval()
        del sd

        self.whisper = WhisperModel.from_pretrained(
            os.path.join(weights_dir, "whisper")
        ).to(self.device, self.dtype)
        self.whisper.eval()
        self.fe = AutoFeatureExtractor.from_pretrained(os.path.join(weights_dir, "whisper"))
        self.pe = PositionalEncoding(384).to(self.device)
        print(f"[musetalk] core loaded on {self.device} in {time.time() - t0:.1f}s", flush=True)

    # -- portrait preparation ---------------------------------------------

    @torch.no_grad()
    def prepare_portrait(self, base_bgr: np.ndarray, bbox):
        """Cache the face crop latents + feathered lower-face blend mask."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = y2 - y1, x2 - x1
        crop = base_bgr[y1:y2, x1:x2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        self.base_bgr = base_bgr
        self.bbox = (x1, y1, x2, y2)

        # 8-channel latents: [half-masked crop, full crop] (MuseTalk format)
        half = resized.copy()
        half[128:, :] = 0
        lat_a = self._encode(half)
        lat_b = self._encode(resized)
        self.latents = torch.cat([lat_a, lat_b], dim=1)  # (1, 8, 32, 32)

        # Feathered rectangle covering the inpainted lower half of the face.
        mask = np.zeros((h, w), np.float32)
        top = int(h * 0.48)
        side = max(4, int(w * 0.12))
        bottom = max(top + 8, h - max(6, int(h * 0.04)))
        mask[top:bottom, side:w - side] = 1.0
        sigma = max(4.0, h / 8.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        self.mask = mask[..., None]

    @torch.no_grad()
    def _encode(self, img256_bgr: np.ndarray) -> torch.Tensor:
        x = img256_bgr.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, self.dtype)
        return self.vae.encode(x).latent_dist.mode() * self.scaling

    # -- audio features ----------------------------------------------------

    @torch.no_grad()
    def audio_chunks(self, audio_f32: np.ndarray, sample_rate: int, fps: int) -> torch.Tensor:
        """(T, 50, 384) whisper prompts, one row per video frame."""
        wav = resample_to_16k(audio_f32, sample_rate)
        n_samples = int(wav.shape[0])

        seg_len = 30 * AUDIO_SR
        features = []
        for i in range(0, max(n_samples, 1), seg_len):
            seg = wav[i:i + seg_len]
            if seg.size == 0:
                seg = np.zeros(AUDIO_SR, np.float32)
            feat = self.fe(seg, return_tensors="pt", sampling_rate=AUDIO_SR).input_features
            features.append(feat)

        per_frame = 2 * (PAD_LEFT + PAD_RIGHT + 1)
        layers = []
        for feat in features:
            feat = feat.to(self.device, self.dtype)
            hs = self.whisper.encoder(feat, output_hidden_states=True).hidden_states
            layers.append(torch.stack(hs, dim=2))
        wf = torch.cat(layers, dim=1)  # (1, T50, L, 384)
        wf = wf[:, : int(n_samples / AUDIO_SR * AUDIO_FPS)]

        multiplier = AUDIO_FPS / fps
        num_frames = int(math.floor((n_samples / AUDIO_SR) * fps))
        pad = int(math.ceil(multiplier))
        wf = torch.cat([
            torch.zeros_like(wf[:, : pad * PAD_LEFT]),
            wf,
            torch.zeros_like(wf[:, : pad * 3 * PAD_RIGHT]),
        ], dim=1)

        clips = []
        for fi in range(num_frames):
            idx = int(math.floor(fi * multiplier))
            clip = wf[:, idx: idx + per_frame]
            if clip.shape[1] != per_frame:
                pad_rows = per_frame - clip.shape[1]
                if pad_rows > 0:
                    clip = torch.cat([clip, torch.zeros_like(clip[:, :pad_rows])], dim=1)
                else:
                    continue
            clips.append(clip)
        out = torch.cat(clips, dim=0)          # (T, per_frame, L, 384)
        b, c, h, w = out.shape
        return out.reshape(b, c * h, w)        # (T, 50, 384)

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def render_batch(self, audio_prompts: torch.Tensor) -> list[np.ndarray]:
        """One UNet+VAE pass; returns RGB frames (H, W, 3) pasted on the portrait."""
        b = audio_prompts.shape[0]
        prompts = self.pe(audio_prompts.to(self.device, self.dtype))
        latents = self.latents.to(self.device, self.dtype).expand(b, -1, -1, -1)
        timesteps = torch.zeros(b, dtype=torch.long, device=self.device)
        pred = self.unet(latents, timesteps, encoder_hidden_states=prompts).sample
        pred = pred.float() / self.scaling
        recon = self.vae.decode(pred.to(self.dtype)).sample
        recon = (recon / 2 + 0.5).clamp(0, 1)
        recon = (recon.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)

        x1, y1, x2, y2 = self.bbox
        out = []
        for img in recon:
            frame = self.base_bgr.copy()
            gen = cv2.resize(img, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
            region = frame[y1:y2, x1:x2].astype(np.float32)
            blended = region * (1.0 - self.mask) + gen.astype(np.float32) * self.mask
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
