"""SER Terminal — dengar ucapan dari mic, gerakkan servo wajah (Raspberry Pi + PCA9685).

Emosi ucapan dideteksi wav2vec2 SER, diterjemahkan ke 5 ekspresi (senyum, netral,
sedih, marah, kaget), lalu jadi pose 11 servo yang ditulis langsung ke PCA9685 lewat I2C:

    mic -> VAD -> wav2vec2 -> skor ekspresi -> pose 11 servo
        --I2C (GPIO2/GPIO3)--> PCA9685 --> servo

File ini hanya perakit (CLI + menyambung objek). Logikanya ada di:
    pipeline.py     jalankan(): mic -> resample -> VAD -> thread inferensi
    audio.py        PenangkapAudio (sounddevice, non-blocking)
    vad.py          SegmenterVAD (Silero VAD -> jendela ucapan)
    model_emosi.py  PengenalEmosi (wav2vec2 HuggingFace) + peta label -> ekspresi
    pose.py         tabel pose (7 gerakan -> 11 servo) + blend skor -> pose
    pengendali.py   PengendaliEkspresi (model -> pose -> servo)
    servo.py        PengirimServo (I2C ke PCA9685; tanpa --i2c-bus = cetak ke terminal)

Tanpa `--i2c-bus`, pose dicetak ke terminal (uji tanpa hardware).

Jalankan:
    python main.py --i2c-bus 1                              # Raspberry Pi: gerakkan servo
    python main.py                                          # cetak pose saja
    python main.py --list-devices                           # daftar mic
    python main.py --device 2 --rms                         # uji level mic saja
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from konfigurasi import CONF_THRESHOLD, HOP_SEC, MODEL_ID, SMOOTH_ALPHA, WINDOW_SEC
from pengendali import PengendaliEkspresi
from pipeline import jalankan, uji_rms
from servo import ALAMAT_PCA, PengirimServo

log = logging.getLogger("ser")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deteksi emosi ucapan dari mic, gerakkan servo wajah lewat PCA9685 (I2C)."
    )
    servo = p.add_argument_group("servo (PCA9685 via I2C)")
    servo.add_argument(
        "--i2c-bus",
        type=int,
        default=None,
        help="Nomor bus I2C (1 = /dev/i2c-1, pin GPIO2/GPIO3 di Pi); kosong = cetak pose saja.",
    )

    p.add_argument("--device", type=int, default=None, help="Index device input mic.")
    p.add_argument("--list-devices", action="store_true", help="Tampilkan device audio.")
    p.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Penguat amplitudo mic (default 1.0). Naikkan, mis. 4, kalau suara terlalu kecil.",
    )
    p.add_argument("--rms", action="store_true", help="Mode uji: cetak level mic saja.")
    p.add_argument("--model", default=MODEL_ID, help="ID model SER di HuggingFace.")
    p.add_argument(
        "--threshold",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Ambang confidence; di bawahnya pose memudar ke netral (default {CONF_THRESHOLD}).",
    )
    p.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Sensitivitas Silero VAD, 0-1 (kecil = lebih sensitif).",
    )
    p.add_argument(
        "--window",
        type=float,
        default=WINDOW_SEC,
        help=f"Panjang jendela audio yang dinilai, detik (default {WINDOW_SEC}).",
    )
    p.add_argument(
        "--hop",
        type=float,
        default=HOP_SEC,
        help=f"Jarak antar-tebakan, detik. Kecil = lebih responsif (default {HOP_SEC}).",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=SMOOTH_ALPHA,
        help=f"Bobot tebakan baru, 0-1. 1.0 = tanpa smoothing (default {SMOOTH_ALPHA}).",
    )
    p.add_argument("--verbose", action="store_true", help="Log detail (debug).")
    return p.parse_args(argv)


def daftar_device() -> int:
    try:
        import sounddevice as sd

        print(sd.query_devices())
    except Exception as exc:
        log.error("Gagal membaca device audio: %s", exc)
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.list_devices:
        return daftar_device()

    if args.rms:  # uji mic saja, servo tidak disentuh
        return uji_rms(args)

    servo = PengirimServo(args.i2c_bus)  # bus kosong = cetak pose saja
    try:
        servo.buka()
    except ImportError:
        log.error("smbus2 belum terpasang: pip install smbus2")
        return 1
    except FileNotFoundError:
        log.error("/dev/i2c-%d tidak ada: aktifkan I2C (sudo raspi-config -> Interface Options -> I2C) lalu reboot.", args.i2c_bus)
        return 1
    except PermissionError:
        log.error("Tidak boleh membuka /dev/i2c-%d: sudo usermod -aG i2c $USER, lalu login ulang.", args.i2c_bus)
        return 1
    except OSError as exc:
        log.error(
            "PCA9685 (0x%02X) tidak menjawab di bus %d: %s. Cek kabel SDA/SCL/GND/VCC dan jalankan: i2cdetect -y %d",
            ALAMAT_PCA, args.i2c_bus, exc, args.i2c_bus,
        )
        return 1

    try:
        pengendali = PengendaliEkspresi(servo, args.threshold, args.smooth)
        return jalankan(args, pengendali.loop)
    finally:
        servo.tutup()  # servo: kirim netral lalu tutup port


if __name__ == "__main__":
    raise SystemExit(main())
