"""Penangkapan audio dari mic (non-blocking) + resample ke 16 kHz."""

from __future__ import annotations

import logging
import queue
from typing import Any, Callable, Optional

import numpy as np

from konfigurasi import SAMPLE_RATE

log = logging.getLogger("ser.audio")


class PenangkapAudio:
    """Buka input stream sounddevice dan lempar blok audio ke antrian.

    Callback sengaja dibuat sangat ringan (hanya copy + put) agar tidak pernah
    mem-block PortAudio. Resample dan VAD dikerjakan di thread lain.
    """

    def __init__(self, device: Optional[int], antrian: queue.Queue) -> None:
        import sounddevice as sd

        self._sd = sd
        self._antrian = antrian
        self.device = device
        self.samplerate = self._pilih_samplerate(device)
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=int(self.samplerate * 0.032),
            callback=self._callback,
        )

    def _pilih_samplerate(self, device: Optional[int]) -> int:
        """Pakai 16 kHz kalau device sanggup; kalau tidak, pakai rate bawaannya."""
        try:
            self._sd.check_input_settings(
                device=device, channels=1, samplerate=SAMPLE_RATE, dtype="float32"
            )
            return SAMPLE_RATE
        except Exception:
            info = self._sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
            log.info("Device tidak mendukung 16 kHz, memakai %d Hz + resample.", rate)
            return rate

    def _callback(self, indata: np.ndarray, frames: int, waktu: Any, status: Any) -> None:
        if status:
            log.debug("Status stream: %s", status)
        self._antrian.put(indata[:, 0].copy())

    def __enter__(self) -> "PenangkapAudio":
        self.stream.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stream.stop()
        self.stream.close()


def buat_resampler(rate_sumber: int) -> Callable[[np.ndarray], np.ndarray]:
    """Kembalikan fungsi resample ke 16 kHz (identitas kalau rate sudah pas)."""
    if rate_sumber == SAMPLE_RATE:
        return lambda blok: blok

    import torch
    import torchaudio.functional as AF

    def resample(blok: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(blok).unsqueeze(0)
        keluar = AF.resample(tensor, rate_sumber, SAMPLE_RATE)
        return keluar.squeeze(0).numpy()

    return resample
