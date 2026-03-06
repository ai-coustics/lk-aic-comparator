import argparse
import os
from pathlib import Path
from types import ModuleType

import aic_sdk as aic
import numpy as np
import soundfile as sf


MODEL_TO_FFI_ENUM = {
    "quail_l": "QUAIL_L",
    "quail_vf_l": "QUAIL_VF_L",
    "sparrow_s": "SPARROW_S",
}

MODEL_TO_DOWNLOAD_ID = {
    "quail_l": "quail-l-16khz",
    "quail_vf_l": "quail-vf-2.0-l-16khz",
    "sparrow_s": "sparrow-s-48khz",
}


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            len(value) >= 2
            and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'"))
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare output from direct aic-sdk processor vs public "
            "livekit-plugins-ai-coustics enhancer."
        )
    )
    parser.add_argument("input_audio", help="Input wav/flac/etc (mono expected)")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_TO_DOWNLOAD_ID),
        default="quail_l",
        help="Model to use in both processors",
    )
    parser.add_argument(
        "--model-download-dir",
        default="models",
        help="Directory to cache downloaded aic-sdk models",
    )
    parser.add_argument(
        "--livekit-room",
        default="lk-aic-comparator",
        help="Room name used when generating a LiveKit access token for plugin auth",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file to load before reading credentials (default: .env)",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=10,
        help="Frame size in milliseconds (plugin expects 10ms in production)",
    )
    parser.add_argument(
        "--atol",
        type=int,
        default=0,
        help="Absolute int16 tolerance for considering samples equal",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for considering samples equal",
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="Write processed outputs next to input for debugging",
    )
    return parser.parse_args()


def _load_ffi_module() -> ModuleType:
    from livekit.plugins.ai_coustics import _ffi

    return _ffi


def _get_livekit_credentials(room_name: str) -> tuple[str, str]:
    livekit_url = os.getenv("LIVEKIT_URL")
    if not livekit_url:
        raise RuntimeError(
            "LIVEKIT_URL is required when using public livekit-plugins-ai-coustics"
        )

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET")

    from livekit import api

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("lk-aic-comparator")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    return livekit_url, token


def _load_mono_float_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        if audio.shape[1] != 1:
            raise ValueError(
                f"Expected mono input, got {audio.shape[1]} channels: {path}"
            )
        audio = audio[:, 0]
    return audio, sample_rate


def _chunk_audio(audio: np.ndarray, frame_samples: int):
    n = audio.shape[0]
    for start in range(0, n, frame_samples):
        end = min(start + frame_samples, n)
        chunk = audio[start:end]
        if chunk.shape[0] < frame_samples:
            padded = np.zeros((frame_samples,), dtype=np.float32)
            padded[: chunk.shape[0]] = chunk
            yield padded, chunk.shape[0]
        else:
            yield chunk, frame_samples


def _process_with_aic_sdk(
    *,
    audio: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    model_name: str,
    model_download_dir: Path,
    license_key: str,
) -> np.ndarray:
    model_id = MODEL_TO_DOWNLOAD_ID[model_name]
    model_path = aic.Model.download(model_id, str(model_download_dir))
    model = aic.Model.from_file(model_path)
    config = aic.ProcessorConfig(
        sample_rate=sample_rate,
        num_channels=1,
        num_frames=frame_samples,
        allow_variable_frames=False,
    )
    processor = aic.Processor(model, license_key, config)

    chunks: list[np.ndarray] = []
    for chunk, valid_len in _chunk_audio(audio, frame_samples):
        planar = chunk.reshape(1, -1).astype(np.float32, copy=False)
        out = processor.process(planar)[0]
        out_i16 = (np.clip(out, -1.0, 1.0) * 32767.0).astype(np.int16)
        chunks.append(out_i16[:valid_len])
    return np.concatenate(chunks, axis=0)


def _process_with_plugin_ffi(
    *,
    ffi: ModuleType,
    audio: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    model_name: str,
    livekit_url: str,
    livekit_token: str,
) -> np.ndarray:
    model_enum = getattr(ffi.EnhancerModel, MODEL_TO_FFI_ENUM[model_name])
    settings = ffi.EnhancerSettings(
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=frame_samples,
        credentials=ffi.Credentials(url=livekit_url, token=livekit_token),
        model=model_enum,
        vad=ffi.VadSettings(
            speech_hold_duration=None,
            sensitivity=None,
            minimum_speech_duration=None,
        ),
    )
    enhancer = ffi.Enhancer(settings)
    enhancer.update_stream_info(
        ffi.StreamInfo(
            room_id="",
            room_name="local-room",
            participant_identity="local-participant",
            participant_id="",
            track_id="local-track",
        )
    )

    chunks: list[np.ndarray] = []
    for chunk, valid_len in _chunk_audio(audio, frame_samples):
        buf = chunk.astype(np.float32, copy=True)
        native = ffi.NativeAudioBufferMut(ptr=buf.ctypes.data, len=buf.shape[0])
        enhancer.process(native)
        out_i16 = (np.clip(buf, -1.0, 1.0) * 32767.0).astype(np.int16)
        chunks.append(out_i16[:valid_len])
    return np.concatenate(chunks, axis=0)


def _print_report(
    sdk_out: np.ndarray,
    plugin_out: np.ndarray,
    *,
    atol: int,
    rtol: float,
) -> None:
    if sdk_out.shape != plugin_out.shape:
        print(f"FAILED: shape mismatch sdk={sdk_out.shape} plugin={plugin_out.shape}")
        return

    diff = sdk_out.astype(np.int32) - plugin_out.astype(np.int32)
    abs_diff = np.abs(diff)
    equal_mask = np.isclose(sdk_out, plugin_out, rtol=rtol, atol=atol)
    mismatches = np.where(~equal_mask)[0]

    print(f"total_samples: {sdk_out.size}")
    print(f"matching_samples: {int(equal_mask.sum())}")
    print(f"mismatching_samples: {int(mismatches.size)}")
    print(f"max_abs_diff: {int(abs_diff.max(initial=0))}")
    print(f"mean_abs_diff: {float(abs_diff.mean()) if abs_diff.size else 0.0:.6f}")

    if mismatches.size == 0:
        print("MATCH: outputs are identical within tolerance")
        return

    first = int(mismatches[0])
    print(
        "FIRST_MISMATCH: "
        f"index={first} sdk={int(sdk_out[first])} plugin={int(plugin_out[first])} "
        f"abs_diff={int(abs_diff[first])}"
    )
    print("MISMATCH: outputs differ")


def main() -> int:
    args = _parse_args()
    _load_env_file(Path(args.env_file).expanduser().resolve())

    input_path = Path(args.input_audio).expanduser().resolve()
    model_download_dir = Path(args.model_download_dir).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    license_key = os.getenv("AIC_SDK_LICENSE")
    if not license_key:
        raise RuntimeError("AIC_SDK_LICENSE is required for direct aic-sdk processing")

    audio, sample_rate = _load_mono_float_audio(input_path)
    frame_samples = sample_rate * args.frame_ms // 1000
    if frame_samples <= 0:
        raise ValueError("Computed frame size is invalid")

    ffi = _load_ffi_module()
    livekit_url, livekit_token = _get_livekit_credentials(args.livekit_room)

    sdk_out = _process_with_aic_sdk(
        audio=audio,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        model_name=args.model,
        model_download_dir=model_download_dir,
        license_key=license_key,
    )
    plugin_out = _process_with_plugin_ffi(
        ffi=ffi,
        audio=audio,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        model_name=args.model,
        livekit_url=livekit_url,
        livekit_token=livekit_token,
    )

    _print_report(sdk_out, plugin_out, atol=args.atol, rtol=args.rtol)

    if args.write_outputs:
        out_dir = input_path.parent
        sf.write(out_dir / f"{input_path.stem}.sdk.wav", sdk_out, sample_rate)
        sf.write(out_dir / f"{input_path.stem}.plugin.wav", plugin_out, sample_rate)
        print(f"wrote: {out_dir / (input_path.stem + '.sdk.wav')}")
        print(f"wrote: {out_dir / (input_path.stem + '.plugin.wav')}")

    return 0
