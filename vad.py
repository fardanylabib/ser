"""Segmentasi ucapan berbasis Silero VAD."""

from collections import deque
from typing import Any, Deque, Optional

import numpy as np

from konfigurasi import (
    HOP_SEC,
    MIN_SPEECH_SEC,
    PREROLL_SEC,
    SAMPLE_RATE,
    VAD_FRAME,
    WINDOW_SEC,
)


class SegmenterVAD:
    """Ubah aliran frame 512 sampel jadi jendela geser audio siap-inferensi."""

    def __init__(
        self,
        vad: Any,
        min_sec: float = MIN_SPEECH_SEC,
        window_sec: float = WINDOW_SEC,
        hop_sec: float = HOP_SEC,
        preroll_sec: float = PREROLL_SEC,
    ) -> None:
        self.vad = vad
        self.min_sampel = int(min_sec * SAMPLE_RATE)
        self.maks_sampel = int(window_sec * SAMPLE_RATE)
        self.hop_sampel = int(hop_sec * SAMPLE_RATE)
        self.preroll: Deque[np.ndarray] = deque(
            maxlen=max(1, int(preroll_sec * SAMPLE_RATE / VAD_FRAME))
        )
        self.buffer: Deque[np.ndarray] = deque()
        self.panjang = 0
        self.sejak_tebak = 0
        self.sedang_bicara = False

    def proses(self, frame: np.ndarray) -> tuple[bool, Optional[np.ndarray], bool]:
        """Proses satu frame; kembalikan (mulai, audio atau None, selesai)."""
        mulai = False
        selesai = False
        audio: Optional[np.ndarray] = None

        if self.sedang_bicara:
            self._tambah(frame)
        else:
            self.preroll.append(frame)

        hasil = self.vad(frame.copy(), return_seconds=False)

        if hasil and "start" in hasil and not self.sedang_bicara:
            self.sedang_bicara = True
            # Sertakan pre-roll agar awal kata utuh.
            self.buffer.clear()
            self.panjang = 0
            self.sejak_tebak = 0
            for f in self.preroll:
                self._tambah(f)
            self.preroll.clear()
            mulai = True

        elif hasil and "end" in hasil and self.sedang_bicara:
            # Tebakan terakhir frasa ini, lalu tutup.
            audio = self._jendela()
            self.sedang_bicara = False
            self.buffer.clear()
            self.panjang = 0
            selesai = True

        elif self.sedang_bicara and self.sejak_tebak >= self.hop_sampel:
            audio = self._jendela()
            self.sejak_tebak = 0

        return mulai, audio, selesai

    def _tambah(self, frame: np.ndarray) -> None:
        """Masukkan frame ke buffer, buang yang keluar dari jendela."""
        self.buffer.append(frame)
        self.panjang += len(frame)
        self.sejak_tebak += len(frame)
        while self.panjang - len(self.buffer[0]) >= self.maks_sampel:
            self.panjang -= len(self.buffer.popleft())

    def _jendela(self) -> Optional[np.ndarray]:
        """Salinan isi buffer, atau None kalau belum cukup panjang."""
        if self.panjang < self.min_sampel:
            return None  # terlalu pendek
        return np.concatenate(self.buffer)

    def selesaikan(self) -> Optional[np.ndarray]:
        """Tutup ucapan yang masih menggantung."""
        if not self.sedang_bicara:
            return None
        self.sedang_bicara = False
        return self._jendela()
