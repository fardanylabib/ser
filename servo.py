"""Tujuan pose: servo lewat ESP32 (UART), atau cetak ke terminal kalau tanpa port.

    PC --USB/UART--> ESP32 --I2C--> PCA9685 --> servo

Sisi ESP32 -> PCA9685 dibuat sendiri; modul ini hanya mengirim teks.

Protokol UART (ASCII, 115200 baud 8N1, satu baris per pose, diakhiri '\\n'):

    E,<buka mata>,<angkat alis>,<mulut x>,<mulut y>\\n        contoh: E,-3,1,8,3

    nilai  : bilangan bulat -10..10 (0 = netral); `mulut y` 0..10 (0 = mingkem)
    buka mata   -10 merem .. 10 melotot
    angkat alis -10 mengernyit .. 10 naik
    mulut x     -10 menguncup .. 10 senyum lebar
    mulut y       0 mingkem .. 10 menganga

Pose berubah langsung tanpa interpolasi; haluskan gerak servo di firmware ESP32.
Baris dikirim saat pose berubah dan diulang tiap KEEPALIVE_SEC walau tetap,
supaya ESP32 bisa mendeteksi PC putus. Saat ditutup dikirim `E,0,0,0,0` (netral).
ESP32 tidak perlu membalas apa pun.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pose

log = logging.getLogger("ser.servo")

BAUD_DEFAULT = 115200
"""Baud rate UART (samakan dengan `Serial.begin(...)` di ESP32)."""

KEEPALIVE_SEC = 0.5
"""Pose diulang sekurang-kurangnya tiap ini walau tidak berubah."""

BOOT_ESP32_SEC = 2.0
"""Jeda setelah port dibuka: membuka port me-reset ESP32 (DTR/RTS), tunggu boot."""

RECONNECT_SEC = 2.0
"""Selang percobaan buka ulang port kalau kabel USB sempat tercabut."""


def ke_nilai(p: dict[str, float]) -> tuple[int, ...]:
    """Bulatkan dan jepit tiap DOF ke batasnya, urutan sesuai `pose.DOF`."""
    return tuple(int(min(pose.BATAS[d][1], max(pose.BATAS[d][0], round(p[d])))) for d in pose.DOF)


class PengirimServo:
    """Kirim pose ke ESP32 lewat port serial.

    Kalau `port` kosong (None), tidak ada servo: pose hanya dicetak ke terminal
    tiap kali berubah (uji tanpa hardware).

    `kirim` dipanggil berkala oleh thread inferensi (tiap ~0,2 dtk), jadi tidak
    perlu thread sendiri: ia menulis kalau pose berubah atau sudah waktunya
    keepalive. Kalau kabel tercabut, port dibuka ulang di panggilan berikutnya.
    """

    def __init__(self, port: Optional[str] = None, baud: int = BAUD_DEFAULT) -> None:
        self.port, self.baud = port, baud
        self._terakhir_cetak: Optional[str] = None  # mode cetak: baris terakhir
        self._ser: Any = None
        self._terkirim: Optional[tuple[int, ...]] = None  # pose terakhir yang ditulis
        self._waktu_kirim = 0.0
        self._siap = 0.0  # baru boleh menulis setelah waktu ini (tunggu ESP32 boot)
        self._coba_lagi = 0.0  # baru boleh menyambung ulang setelah waktu ini

    def buka(self) -> None:
        """Buka port; lempar exception kalau gagal (agar program berhenti di awal)."""
        if self.port is None:  # mode cetak
            return
        import serial  # pyserial; diimpor di sini agar mode tanpa servo tidak butuh

        self._ser = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0.5)
        self._terkirim = None
        self._siap = time.monotonic() + BOOT_ESP32_SEC
        log.info("Servo: terhubung ke %s @ %d baud", self.port, self.baud)

    def kirim(self, p: dict[str, float]) -> None:
        if self.port is None:  # mode cetak: hanya kalau pose berubah
            baris = pose.format_pose(p)
            if baris != self._terakhir_cetak:
                self._terakhir_cetak = baris
                print(baris, flush=True)
            return

        nilai = ke_nilai(p)
        sekarang = time.monotonic()

        if self._ser is None:  # port lepas: coba sambung lagi tiap RECONNECT_SEC
            if sekarang < self._coba_lagi:
                return
            try:
                self.buka()
            except Exception as exc:
                self._coba_lagi = sekarang + RECONNECT_SEC
                log.debug("Servo: gagal membuka %s: %s", self.port, exc)
                return

        if sekarang < self._siap:
            return
        if nilai == self._terkirim and sekarang - self._waktu_kirim < KEEPALIVE_SEC:
            return
        try:
            self._tulis(nilai)
        except Exception as exc:  # kabel dicabut / ESP32 hilang
            log.warning("Servo: koneksi terputus (%s); mencoba menyambung ulang", exc)
            self._lepas()
            self._coba_lagi = sekarang + RECONNECT_SEC

    def tutup(self) -> None:
        """Kirim pose netral lalu tutup port."""
        if self._ser is not None:
            try:
                self._tulis(ke_nilai(pose.NETRAL))
            except Exception:
                pass
        self._lepas()

    def _tulis(self, nilai: tuple[int, ...]) -> None:
        self._ser.write(("E," + ",".join(map(str, nilai)) + "\n").encode("ascii"))
        self._terkirim, self._waktu_kirim = nilai, time.monotonic()

    def _lepas(self) -> None:
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None
