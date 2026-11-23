"""Pengirim pose ke servo: Raspberry Pi -> PCA9685 lewat I2C (tanpa perantara).

    Raspberry Pi --I2C (GPIO2 SDA, GPIO3 SCL)--> PCA9685 --> servo

Pi menulis register PCA9685 langsung lewat `/dev/i2c-N` (library `smbus2`).
Semua yang perlu diketahui tentang PCA9685 (alamat, register, inisialisasi,
50 Hz, kalibrasi servo) ada di file ini.

Servo ke-n (urutan `pose.SERVO`) dipasang di channel PCA9685 ke-(n-1).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pose

log = logging.getLogger("ser.servo")

RECONNECT_SEC = 2.0
"""Selang percobaan ulang kalau PCA9685 tidak menjawab (kabel lepas / catu mati)."""

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


def data_pose(tick: tuple[int, ...]) -> list[int]:
    """Isi tulis beruntun ke register LED0..: tiap channel 4 byte (ON_L, ON_H, OFF_L, OFF_H)."""
    isi: list[int] = []
    for t in tick:
        isi += [0, 0, t & 0xFF, t >> 8]  # ON di tick 0, OFF di tick `t` -> lebar pulsa = t
    return isi


class PengirimServo:
    """Kirim pose ke PCA9685 lewat I2C Raspberry Pi.

    `kirim` dipanggil berkala oleh thread inferensi (tiap ~0,2 dtk), jadi tidak
    perlu thread sendiri: ia menulis hanya kalau pose berubah. Kalau PCA9685
    berhenti menjawab, penulisan dicoba lagi (dan PCA9685 diinisialisasi ulang)
    tiap RECONNECT_SEC.

    Kalau `bus` kosong (None), tidak ada servo: pose hanya dicetak ke terminal
    tiap kali berubah (uji tanpa hardware).
    """

    def __init__(self, bus: Optional[int] = None) -> None:
        self.bus = bus  # nomor bus I2C: 1 -> /dev/i2c-1 (pin GPIO2/GPIO3 di Pi)
        self._smbus: Any = None
        self._i2c_msg: Any = None  # modul smbus2.i2c_msg (diimpor di buka())
        self._terkirim: Optional[tuple[int, ...]] = None  # tick terakhir yang ditulis
        self._siap = False  # PCA9685 sudah diinisialisasi dan menjawab
        self._putus = False  # sedang dalam kondisi gagal (agar peringatan tidak berulang)
        self._coba_lagi = 0.0  # baru boleh mencoba ulang setelah waktu ini
        self._terakhir_cetak: Optional[str] = None  # mode cetak: baris terakhir

    def buka(self) -> None:
        """Buka bus I2C dan inisialisasi PCA9685; lempar exception kalau gagal.

        Dibuat gagal-cepat agar salah kabel / I2C belum aktif ketahuan di awal,
        bukan baru saat ucapan pertama.
        """
        if self.bus is None:  # mode cetak
            return
        from smbus2 import SMBus, i2c_msg  # diimpor di sini agar mode cetak tidak butuh

        self._i2c_msg = i2c_msg
        self._smbus = SMBus(self.bus)
        try:
            self._init_pca()
        except Exception:
            self._lepas()
            raise
        log.info("Servo: PCA9685 0x%02X terhubung di /dev/i2c-%d", ALAMAT_PCA, self.bus)

    def _tulis(self, reg: int, data: list[int]) -> None:
        """Satu transaksi I2C: [register][data...] ke PCA9685 (tanpa batas 32 byte SMBus)."""
        self._smbus.i2c_rdwr(self._i2c_msg.write(ALAMAT_PCA, [reg, *data]))

    def _init_pca(self) -> None:
        """Urutan inisialisasi PCA9685; dipanggil tiap kali (re)connect."""
        self._tulis(REG_MODE1, [0x10])  # tidur (PRESCALE hanya bisa diubah saat tidur)
        self._tulis(REG_PRESCALE, [PRESCALE])  # 50 Hz
        self._tulis(REG_MODE2, [0x04])  # keluaran totem-pole
        self._tulis(REG_MODE1, [0x20])  # bangun + auto-increment register
        time.sleep(0.01)  # osilator PCA9685 butuh waktu setelah bangun
        self._siap = True
        self._terkirim = None  # PCA9685 baru di-reset -> pose harus ditulis ulang

    def kirim(self, p: dict[str, float]) -> None:
        if self.bus is None:  # mode cetak: hanya kalau pose berubah
            baris = pose.format_pose(p)
            if baris != self._terakhir_cetak:
                self._terakhir_cetak = baris
                print(baris, flush=True)
            return

        tick = ke_tick(p)
        sekarang = time.monotonic()
        if sekarang < self._coba_lagi:
            return
        try:
            if not self._siap:
                self._init_pca()
                log.info("Servo: PCA9685 menjawab lagi")
                self._putus = False
            if tick != self._terkirim:
                self._tulis(REG_LED0_ON_L, data_pose(tick))
                self._terkirim = tick
        except OSError as exc:  # PCA9685 tidak menjawab: kabel lepas / catu mati
            if not self._putus:
                log.warning("Servo: PCA9685 tidak menjawab (%s); mencoba lagi tiap %.0f dtk", exc, RECONNECT_SEC)
                self._putus = True
            self._siap = False
            self._coba_lagi = sekarang + RECONNECT_SEC

    def tutup(self) -> None:
        """Kembalikan semua servo ke netral lalu tutup bus."""
        if self._smbus is not None and self._siap:
            try:
                self._tulis(REG_LED0_ON_L, data_pose(ke_tick(pose.NETRAL)))
            except OSError:
                pass
        self._lepas()

    def _lepas(self) -> None:
        try:
            if self._smbus is not None:
                self._smbus.close()
        except Exception:
            pass
        self._smbus = None
        self._siap = False
