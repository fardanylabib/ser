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
            # Buffer PortAudio dibuat besar. Default "low" terlalu tipis untuk mic USB
            # di Raspberry Pi: sedikit telat membaca saja sudah "input overflow" dan
            # audio bolong. Imbalannya hanya jeda tangkap sekian puluh ms.
            latency="high",
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


def terapkan_gain(blok: np.ndarray, gain: float) -> np.ndarray:
    """Kalikan amplitudo dengan `gain` lalu potong ke [-1, 1] (gain 1.0 = tidak diubah).

    Untuk mic yang levelnya kecil padahal gain ALSA sudah maksimum.
    """
    if gain == 1.0:
        return blok
    return np.clip(blok * gain, -1.0, 1.0)


def buat_resampler(rate_sumber: int) -> Callable[[np.ndarray], np.ndarray]:
    """Kembalikan fungsi resample ke 16 kHz (identitas kalau rate sudah pas)."""
    if rate_sumber == SAMPLE_RATE:
        return lambda blok: blok

    import torch
    import torchaudio.transforms as T

    # Kernel filter dihitung SEKALI di sini. `torchaudio.functional.resample`
    # menghitung ulang kernel di tiap panggilan; untuk blok 32 ms (31x/detik)
    # itu ~8x lebih lambat dan di Raspberry Pi bisa membuat pembacaan mic
    # tertinggal (log "input overflow"). Hasil audionya sama.
    pengubah = T.Resample(rate_sumber, SAMPLE_RATE)

    def resample(blok: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return pengubah(torch.from_numpy(blok).unsqueeze(0)).squeeze(0).numpy()

    return resample
