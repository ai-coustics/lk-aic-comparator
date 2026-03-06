import argparse
import importlib.util
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare output from direct aic-sdk processor vs local "
            "livekit-plugins-ai-coustics FFI enhancer."
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
        "--plugin-package-dir",
        default=(
            "~/dev/projects/plugins-ai-coustics-internal/target/packages/python/"
            "src/livekit/plugins/ai_coustics"
        ),
        help="Path that contains _ffi.py and libplugins_ai_coustics_uniffi.so",
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


def _load_ffi_module(plugin_package_dir: Path) -> ModuleType:
    ffi_path = plugin_package_dir / "_ffi.py"
    if not ffi_path.is_file():
        raise FileNotFoundError(f"FFI module not found: {ffi_path}")

    spec = importlib.util.spec_from_file_location("lk_aic_ffi", ffi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {ffi_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
) -> np.ndarray:
    model_enum = getattr(ffi.EnhancerModel, MODEL_TO_FFI_ENUM[model_name])
    settings = ffi.EnhancerSettings(
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=frame_samples,
        credentials=ffi.Credentials(url="https://local.test", token="local-test-token"),
        model=model_enum,
        model_parameters={},
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
    input_path = Path(args.input_audio).expanduser().resolve()
    model_download_dir = Path(args.model_download_dir).expanduser().resolve()
    plugin_package_dir = Path(args.plugin_package_dir).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    license_key = os.getenv("AIC_SDK_LICENSE")
    if not license_key:
        raise RuntimeError("AIC_SDK_LICENSE is required for direct aic-sdk processing")

    audio, sample_rate = _load_mono_float_audio(input_path)
    frame_samples = sample_rate * args.frame_ms // 1000
    if frame_samples <= 0:
        raise ValueError("Computed frame size is invalid")

    ffi = _load_ffi_module(plugin_package_dir)

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
    )

    _print_report(sdk_out, plugin_out, atol=args.atol, rtol=args.rtol)

    if args.write_outputs:
        out_dir = input_path.parent
        sf.write(out_dir / f"{input_path.stem}.sdk.wav", sdk_out, sample_rate)
        sf.write(out_dir / f"{input_path.stem}.plugin.wav", plugin_out, sample_rate)
        print(f"wrote: {out_dir / (input_path.stem + '.sdk.wav')}")
        print(f"wrote: {out_dir / (input_path.stem + '.plugin.wav')}")

    return 0
