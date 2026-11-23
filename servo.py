"""Pengirim pose ke servo: PC -> ESP32 (UART) -> PCA9685 (I2C).

    PC --USB/UART--> ESP32 --I2C--> PCA9685 --> servo

ESP32 hanyalah JEMBATAN: ia tidak mengolah data sedikit pun. Semua yang perlu
diketahui tentang PCA9685 (alamat, register, inisialisasi, 50 Hz, kalibrasi servo)
ada di file ini, di sisi PC. PC mengirim perintah I2C yang sudah jadi, ESP32
meneruskannya apa adanya.

Bingkai UART (biner, 115200 baud 8N1):

    [N] [byte 1] [byte 2] ... [byte N]

    N        1 byte: jumlah byte yang menyusul (1..255)
    byte 1   alamat I2C tujuan (PCA9685 = 0x40)
    byte 2.. isi yang ditulis ke alamat itu (byte pertamanya nomor register)

Tugas firmware ESP32, seluruhnya:

    1. baca 1 byte -> N
    2. baca N byte berikutnya ke buffer
    3. Wire.beginTransmission(buffer[0]); Wire.write(buffer + 1, N - 1); Wire.endTransmission();
    4. ulangi

Tanpa parsing teks, tanpa tabel, tanpa konversi, tanpa inisialisasi PCA9685.
ESP32 tidak perlu membalas apa pun.

Servo ke-n (urutan `pose.SERVO`) dipasang di channel PCA9685 ke-(n-1).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pose

log = logging.getLogger("ser.servo")

BAUD_DEFAULT = 115200
"""Baud rate UART (samakan dengan `Serial.begin(...)` di ESP32)."""

BOOT_ESP32_SEC = 2.0
"""Jeda setelah port dibuka: membuka port me-reset ESP32 (DTR/RTS), tunggu boot."""

RECONNECT_SEC = 2.0
"""Selang percobaan buka ulang port kalau kabel USB sempat tercabut."""

# --- PCA9685 ---------------------------------------------------------------- #

ALAMAT_PCA = 0x40
"""Alamat I2C PCA9685 (default board, semua jumper alamat terbuka)."""

REG_MODE1, REG_MODE2, REG_LED0_ON_L, REG_PRESCALE = 0x00, 0x01, 0x06, 0xFE

PWM_HZ = 50
"""Frekuensi PWM servo analog standar: periode 20 ms = 4096 tick."""

PRESCALE = round(25_000_000 / (4096 * PWM_HZ)) - 1
"""Nilai register PRESCALE untuk PWM_HZ (osilator internal 25 MHz) -> 121 untuk 50 Hz."""

# Kalibrasi: (tick pada nilai terendah, tick pada nilai tertinggi) tiap servo.
# 1 tick = 20 ms / 4096 ~ 4,9 µs, jadi 205 tick ~ 1,0 ms, 307 ~ 1,5 ms, 410 ~ 2,0 ms.
# Nilai terendah/tertinggi = BATAS gerakan sumbernya (-10..10; mulut buka 0..10).
# Isi awal 1,0-2,0 ms yang aman untuk servo hobi umum, bukan angka keramat:
#  - sempitkan (mis. 250, 370) kalau servo menabrak batas mekanik,
#  - TUKAR kedua angka (mis. 410, 205) kalau servo terpasang berlawanan arah.
KALIBRASI: dict[str, tuple[int, int]] = {
    "kelopak kiri": (205, 410),
    "kelopak kanan": (205, 410),
    "alis kiri": (205, 410),
    "alis kanan": (205, 410),
    "sudut mulut kiri": (205, 410),
    "sudut mulut kanan": (205, 410),
    "mulut buka": (205, 410),
    "mata kiri x": (205, 410),
    "mata kanan x": (205, 410),
    "mata y": (205, 410),
    "sudut mata": (205, 410),
}


def ke_tick(p: dict[str, float]) -> tuple[int, ...]:
    """Ubah pose (7 gerakan, -10..10) jadi tick PWM untuk 11 servo (urutan `pose.SERVO`)."""
    hasil = []
    for nama, dof in pose.SERVO:
        lo, hi = pose.BATAS[dof]
        nilai = min(float(hi), max(float(lo), p[dof]))
        tick_lo, tick_hi = KALIBRASI[nama]
        tick = tick_lo + (nilai - lo) / (hi - lo) * (tick_hi - tick_lo)
        hasil.append(min(4095, max(0, round(tick))))
    return tuple(hasil)


def bingkai_i2c(*isi: int) -> bytes:
    """Bungkus satu penulisan I2C ke PCA9685 jadi bingkai UART: [N][alamat][isi...]."""
    badan = bytes([ALAMAT_PCA, *isi])
    return bytes([len(badan)]) + badan


def bingkai_pose(tick: tuple[int, ...]) -> bytes:
    """Satu penulisan beruntun ke register LED0..: tiap channel 4 byte (ON_L, ON_H, OFF_L, OFF_H)."""
    isi = [REG_LED0_ON_L]
    for t in tick:
        isi += [0, 0, t & 0xFF, t >> 8]  # ON di tick 0, OFF di tick `t` -> lebar pulsa = t
    return bingkai_i2c(*isi)


BINGKAI_INIT = (
    bingkai_i2c(REG_MODE1, 0x10),  # tidur (PRESCALE hanya bisa diubah saat tidur)
    bingkai_i2c(REG_PRESCALE, PRESCALE),  # 50 Hz
    bingkai_i2c(REG_MODE2, 0x04),  # keluaran totem-pole
    bingkai_i2c(REG_MODE1, 0x20),  # bangun + auto-increment register
)
"""Urutan inisialisasi PCA9685, dikirim sekali tiap kali port tersambung."""


class PengirimServo:
    """Kirim pose ke PCA9685 lewat ESP32.

    `kirim` dipanggil berkala oleh thread inferensi (tiap ~0,2 dtk), jadi tidak
    perlu thread sendiri: ia menulis hanya kalau pose berubah. Kalau kabel
    tercabut, port dibuka ulang (dan PCA9685 diinisialisasi ulang) di panggilan
    berikutnya.

    Kalau `port` kosong (None), tidak ada servo: pose hanya dicetak ke terminal
    tiap kali berubah (uji tanpa hardware).
    """

    def __init__(self, port: Optional[str] = None, baud: int = BAUD_DEFAULT) -> None:
        self.port, self.baud = port, baud
        self._ser: Any = None
        self._terkirim: Optional[tuple[int, ...]] = None  # tick terakhir yang ditulis
        self._perlu_init = False  # PCA9685 belum diinisialisasi sejak port dibuka
        self._siap = 0.0  # baru boleh menulis setelah waktu ini (tunggu ESP32 boot)
        self._coba_lagi = 0.0  # baru boleh menyambung ulang setelah waktu ini
        self._terakhir_cetak: Optional[str] = None  # mode cetak: baris terakhir

    def buka(self) -> None:
        """Buka port; lempar exception kalau gagal (agar program berhenti di awal)."""
        if self.port is None:  # mode cetak
            return
        import serial  # pyserial; diimpor di sini agar mode tanpa servo tidak butuh

        self._ser = serial.Serial(self.port, self.baud, timeout=0, write_timeout=0.5)
        self._terkirim = None
        self._perlu_init = True
        self._siap = time.monotonic() + BOOT_ESP32_SEC
        log.info("Servo: terhubung ke %s @ %d baud", self.port, self.baud)

    def kirim(self, p: dict[str, float]) -> None:
        if self.port is None:  # mode cetak: hanya kalau pose berubah
            baris = pose.format_pose(p)
            if baris != self._terakhir_cetak:
                self._terakhir_cetak = baris
                print(baris, flush=True)
            return

        tick = ke_tick(p)
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
            return  # ESP32 masih boot; pose dicoba lagi di panggilan berikutnya
        try:
            if self._perlu_init:
                for bingkai in BINGKAI_INIT:
                    self._ser.write(bingkai)
                time.sleep(0.01)  # osilator PCA9685 butuh waktu setelah bangun
                self._perlu_init = False
            if tick != self._terkirim:
                self._ser.write(bingkai_pose(tick))
                self._terkirim = tick
        except Exception as exc:  # kabel dicabut / ESP32 hilang
            log.warning("Servo: koneksi terputus (%s); mencoba menyambung ulang", exc)
            self._lepas()
            self._coba_lagi = sekarang + RECONNECT_SEC

    def tutup(self) -> None:
        """Kembalikan semua servo ke netral lalu tutup port."""
        if self._ser is not None and not self._perlu_init:
            try:
                self._ser.write(bingkai_pose(ke_tick(pose.NETRAL)))
            except Exception:
                pass
        self._lepas()

    def _lepas(self) -> None:
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None
