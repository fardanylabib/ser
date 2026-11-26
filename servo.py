"""Pengirim pose ke servo: PC -> ESP32 (UART, bingkai [N][alamat][isi...]) -> PCA9685 (I2C)."""

import logging
import time
from typing import Any, Optional

import pose

log = logging.getLogger("ser.servo")

BAUD_DEFAULT = 115200
"""Baud rate UART (samakan dengan firmware ESP32)."""

BOOT_ESP32_SEC = 2.0
"""Jeda tunggu ESP32 boot setelah port dibuka."""

RECONNECT_SEC = 2.0
"""Selang percobaan buka ulang port."""

# PCA9685

ALAMAT_PCA = 0x40
"""Alamat I2C PCA9685."""

REG_MODE1, REG_MODE2, REG_LED0_ON_L, REG_PRESCALE = 0x00, 0x01, 0x06, 0xFE

PWM_HZ = 50
"""Frekuensi PWM servo analog standar."""

PRESCALE = round(25_000_000 / (4096 * PWM_HZ)) - 1
"""Nilai register PRESCALE untuk PWM_HZ (osilator 25 MHz)."""

# Kalibrasi (tick di nilai terendah, tick di nilai tertinggi); tukar angka untuk membalik arah.
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
    """Ubah pose (7 gerakan) jadi tick PWM 11 servo."""
    hasil = []
    for nama, dof in pose.SERVO:
        lo, hi = pose.BATAS[dof]
        nilai = min(float(hi), max(float(lo), p[dof]))
        tick_lo, tick_hi = KALIBRASI[nama]
        tick = tick_lo + (nilai - lo) / (hi - lo) * (tick_hi - tick_lo)
        hasil.append(min(4095, max(0, round(tick))))
    return tuple(hasil)


def bingkai_i2c(*isi: int) -> bytes:
    """Bungkus satu penulisan I2C jadi bingkai UART."""
    badan = bytes([ALAMAT_PCA, *isi])
    return bytes([len(badan)]) + badan


def bingkai_pose(tick: tuple[int, ...]) -> bytes:
    """Satu penulisan beruntun ke register LED0.. (4 byte per channel)."""
    isi = [REG_LED0_ON_L]
    for t in tick:
        isi += [0, 0, t & 0xFF, t >> 8]  # ON di 0, OFF di t
    return bingkai_i2c(*isi)


BINGKAI_INIT = (
    bingkai_i2c(REG_MODE1, 0x10),  # tidur
    bingkai_i2c(REG_PRESCALE, PRESCALE),  # 50 Hz
    bingkai_i2c(REG_MODE2, 0x04),  # keluaran totem-pole
    bingkai_i2c(REG_MODE1, 0x20),  # bangun + auto-increment
)
"""Urutan inisialisasi PCA9685 tiap port tersambung."""


class PengirimServo:
    """Kirim pose ke PCA9685 lewat ESP32; tanpa port, pose dicetak ke terminal."""

    def __init__(self, port: Optional[str] = None, baud: int = BAUD_DEFAULT) -> None:
        self.port, self.baud = port, baud
        self._ser: Any = None
        self._terkirim: Optional[tuple[int, ...]] = None  # tick terakhir yang ditulis
        self._perlu_init = False  # PCA9685 belum diinisialisasi
        self._siap = 0.0  # boleh menulis setelah waktu ini
        self._coba_lagi = 0.0  # boleh menyambung ulang setelah waktu ini
        self._terakhir_cetak: Optional[str] = None  # mode cetak: baris terakhir

    def buka(self) -> None:
        """Buka port; lempar exception kalau gagal."""
        if self.port is None:  # mode cetak
            return
        import serial  # diimpor di sini agar mode cetak tidak butuh pyserial

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

        if self._ser is None:  # port lepas: coba sambung lagi
            if sekarang < self._coba_lagi:
                return
            try:
                self.buka()
            except Exception as exc:
                self._coba_lagi = sekarang + RECONNECT_SEC
                log.debug("Servo: gagal membuka %s: %s", self.port, exc)
                return

        if sekarang < self._siap:
            return  # ESP32 masih boot
        try:
            if self._perlu_init:
                for bingkai in BINGKAI_INIT:
                    self._ser.write(bingkai)
                time.sleep(0.01)  # osilator butuh waktu setelah bangun
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
