"""Pengendali ekspresi: hasil model SER -> pose -> servo."""

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

import pose
from konfigurasi import CONF_THRESHOLD, SMOOTH_ALPHA
from model_emosi import PengenalEmosi
from pipeline import AKHIR
from servo import PengirimServo

log = logging.getLogger("ser.pengendali")

IDLE_SEC = 2.5
"""Tanpa tebakan baru selama ini, pose dikembalikan ke netral."""

BOBOT_EKSPRESI: dict[str, float] = {"sedih": 1.8, "netral": 0.7}
"""Koreksi skor per ekspresi (heuristik) sebelum dinormalkan ulang."""


class PengendaliEkspresi:
    """Mengubah jendela audio jadi pose dan mengirimnya ke servo."""

    def __init__(
        self,
        servo: PengirimServo,
        ambang: float = CONF_THRESHOLD,
        alpha: float = SMOOTH_ALPHA,
    ) -> None:
        self.servo = servo
        self.ambang = ambang  # di bawah ini pose memudar ke netral
        self.alpha = alpha  # bobot tebakan baru pada peredam EMA
        self.ema: Optional[np.ndarray] = None  # probabilitas label, sudah diperhalus
        self.pose = dict(pose.NETRAL)  # pose yang sedang terpasang
        self.waktu_tebakan = time.monotonic()

    def loop(
        self,
        pengenal: PengenalEmosi,
        antrian_segmen: queue.Queue,
        berhenti: threading.Event,
    ) -> None:
        """Konsumsi jendela audio sampai `berhenti` atau sentinel None."""
        while not berhenti.is_set():
            try:
                item = antrian_segmen.get(timeout=0.2)
            except queue.Empty:
                self._saat_diam()
                continue
            if item is None:  # sinyal shutdown
                break
            if isinstance(item, str) and item == AKHIR:
                self.ema = None  # frasa baru dinilai segar
                continue
            self._proses(pengenal, item)

    def _proses(self, pengenal: PengenalEmosi, audio: np.ndarray) -> None:
        """Satu jendela audio -> satu pose baru."""
        try:
            probs = pengenal.probabilitas(audio)
        except Exception as exc:  # jangan sampai thread mati gara-gara 1 jendela
            log.error("Inferensi gagal: %s", exc)
            return

        self.ema = probs if self.ema is None else (1.0 - self.alpha) * self.ema + self.alpha * probs
        skor = self._skor_ekspresi(self.ema, pengenal.peta)
        self.pose = pose.hitung_pose(skor, self.ambang)
        self.waktu_tebakan = time.monotonic()

        log.debug("label mentah: %s", {pengenal.labels[i]: round(float(v), 2) for i, v in enumerate(self.ema)})
        log.debug("skor: %s | pose: %s", {e: round(s, 2) for e, s in skor.items()}, pose.format_pose(self.pose))
        self.servo.kirim(self.pose)

    def _saat_diam(self) -> None:
        """Antrean kosong: netral kalau lama diam, lalu kirim ulang pose."""
        if time.monotonic() - self.waktu_tebakan > IDLE_SEC:
            self.pose = dict(pose.NETRAL)
        self.servo.kirim(self.pose)  # servo hanya menulis kalau berubah

    @staticmethod
    def _skor_ekspresi(probs: np.ndarray, peta: list[str]) -> dict[str, float]:
        """Jumlahkan probabilitas label serumpun jadi skor per ekspresi (total = 1)."""
        skor: dict[str, float] = {}
        for i, p in enumerate(probs):
            skor[peta[i]] = skor.get(peta[i], 0.0) + float(p)
        skor = {e: v * BOBOT_EKSPRESI.get(e, 1.0) for e, v in skor.items()}
        total = sum(skor.values()) or 1.0
        return {e: v / total for e, v in skor.items()}
