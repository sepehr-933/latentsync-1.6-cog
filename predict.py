# Cog predictor for LatentSync 1.6 (512x512), built on ByteDance's own code.
#
# Differences from ByteDance's bundled predict.py, which still loads the
# original 256px weights with configs/unet/stage2.yaml:
#   - loads the 1.6 checkpoint (huggingface.co/ByteDance/LatentSync-1.6) with
#     configs/unet/stage2_512.yaml, so the mouth is generated at 512x512;
#   - builds the pipeline ONCE in setup() instead of re-loading the 5 GB UNet
#     in a subprocess on every prediction;
#   - gives every prediction its own output file and temp dir;
#   - exposes inference_steps (ByteDance recommends 20-50; more = sharper).

import os
import subprocess
import time
import uuid

import torch
from accelerate.utils import set_seed
from cog import BasePredictor, Input, Path
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.face_detector import FaceDetector
from latentsync.whisper.audio2feature import Audio2Feature

CONFIG_PATH = "configs/unet/stage2_512.yaml"
HF_BASE = "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main"
WEIGHTS = {
    "checkpoints/latentsync_unet.pt": f"{HF_BASE}/latentsync_unet.pt",
    "checkpoints/whisper/tiny.pt": f"{HF_BASE}/whisper/tiny.pt",
}


def download(url: str, dest: str) -> None:
    start = time.time()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print(f"downloaded in {time.time() - start:.1f}s")


class Predictor(BasePredictor):
    def setup(self) -> None:
        for dest, url in WEIGHTS.items():
            if not os.path.exists(dest):
                download(url, dest)

        self.config = OmegaConf.load(CONFIG_PATH)
        fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        self.dtype = torch.float16 if fp16 else torch.float32

        audio_encoder = Audio2Feature(
            model_path="checkpoints/whisper/tiny.pt",
            device="cuda",
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            "checkpoints/latentsync_unet.pt",
            device="cpu",
        )
        unet = unet.to(dtype=self.dtype)
        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained("configs"),
        ).to("cuda")

        # Fetch InsightFace's detector models now (into checkpoints/auxiliary)
        # so the first prediction does not pay for the download.
        FaceDetector(device="cuda")

    def predict(
        self,
        video: Path = Input(description="Input video (the face to lip-sync)"),
        audio: Path = Input(description="Input audio (the speech to lip-sync to)"),
        guidance_scale: float = Input(
            description="Higher = tighter lip-sync, but may add distortion or jitter",
            ge=1.0,
            le=3.0,
            default=1.5,
        ),
        inference_steps: int = Input(
            description="Denoising steps. Higher = sharper, slower", ge=10, le=50, default=20
        ),
        seed: int = Input(description="Set to 0 for a random seed", default=0),
    ) -> Path:
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        set_seed(seed)

        run_id = uuid.uuid4().hex[:8]
        out_path = f"/tmp/output-{run_id}.mp4"
        self.pipeline(
            video_path=str(video),
            audio_path=str(audio),
            video_out_path=out_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=self.dtype,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask_image_path=self.config.data.mask_image_path,
            temp_dir=f"/tmp/latentsync-{run_id}",
        )
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("LatentSync produced no output video.")
        return Path(out_path)
