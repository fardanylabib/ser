"""Keadaan bersama (thread inferensi -> GUI)."""

from __future__ import annotations

import threading
import time

from constants import IDLE_SEC, NETRAL


class Keadaan:
    """Target pose dari thread inferensi, dibaca thread GUI (dijaga lock)."""

    def __init__(self) -> None:
        self._kunci = threading.Lock()
        self._target = dict(NETRAL)
        self._label = "netral"
        self._conf = 0.0
        self._waktu = 0.0
        self._bicara = False
        self._rms = 0.0
        self._waktu_suara = 0.0

    def set_suara(self, bicara: bool, rms: float) -> None:
        """Dipanggil tiap frame 32 ms: apakah VAD bilang ada ucapan + level RMS-nya."""
        with self._kunci:
            self._bicara, self._rms = bicara, rms
            self._waktu_suara = time.monotonic()

    def baca_suara(self) -> tuple[bool, float]:
        """Kembalikan (sedang bicara, rms); dianggap diam kalau frame berhenti datang."""
        with self._kunci:
            if time.monotonic() - self._waktu_suara > 0.3:
                return False, 0.0
            return self._bicara, self._rms

    def set(self, target: dict[str, float], label: str, conf: float) -> None:
        with self._kunci:
            self._target, self._label, self._conf = target, label, conf
            self._waktu = time.monotonic()

    def baca(self) -> tuple[dict[str, float], str, float]:
        """Kembalikan (target, label, conf); netral kalau sudah lama diam."""
        with self._kunci:
            if time.monotonic() - self._waktu > IDLE_SEC:
                return dict(NETRAL), "netral", 0.0
            return dict(self._target), self._label, self._conf


KEADAAN = Keadaan()
