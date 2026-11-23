"""SER Terminal — dengar ucapan dari mic, gerakkan servo wajah (ESP32 + PCA9685).

Emosi ucapan dideteksi wav2vec2 SER, diterjemahkan ke 5 ekspresi (senyum, netral,
sedih, marah, kaget), lalu jadi pose 4-DOF yang dikirim ke ESP32 lewat USB-serial:

    mic -> VAD -> wav2vec2 -> skor ekspresi -> pose 4-DOF
        --USB/UART--> ESP32 --I2C--> PCA9685 --> servo

File ini hanya perakit (CLI + menyambung objek). Logikanya ada di:
    pipeline.py     jalankan(): mic -> resample -> VAD -> thread inferensi
    audio.py        PenangkapAudio (sounddevice, non-blocking)
    vad.py          SegmenterVAD (Silero VAD -> jendela ucapan)
    model_emosi.py  PengenalEmosi (wav2vec2 HuggingFace) + peta label -> ekspresi
    pose.py         tabel pose 4-DOF + blend skor -> pose
    pengendali.py   PengendaliEkspresi (model -> pose -> servo)
    servo.py        PengirimServo (UART ke ESP32; tanpa port = cetak ke terminal)

Tanpa `--port`, pose dicetak ke terminal (uji tanpa hardware).

Jalankan:
    python main.py --list-ports
    python main.py --port /dev/cu.usbserial-0001           # macOS/Linux
    python main.py --port COM5                              # Windows
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
from servo import BAUD_DEFAULT, PengirimServo

log = logging.getLogger("ser")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deteksi emosi ucapan dari mic, gerakkan servo wajah lewat ESP32."
    )
    servo = p.add_argument_group("servo (ESP32)")
    servo.add_argument("--port", default=None, help="Port serial ESP32; kosong = cetak pose saja.")
    servo.add_argument("--baud", type=int, default=BAUD_DEFAULT, help=f"Baud rate (default {BAUD_DEFAULT}).")
    servo.add_argument("--list-ports", action="store_true", help="Tampilkan port serial lalu keluar.")

    p.add_argument("--device", type=int, default=None, help="Index device input mic.")
    p.add_argument("--list-devices", action="store_true", help="Tampilkan device audio.")
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


def daftar_port() -> int:
    try:
        from serial.tools import list_ports
    except ImportError:
        log.error("pyserial belum terpasang: pip install pyserial")
        return 1
    for info in list_ports.comports():
        print(f"{info.device}\t{info.description}")
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
    if args.list_ports:
        return daftar_port()

    if args.rms:  # uji mic saja, servo tidak disentuh
        return uji_rms(args)

    servo = PengirimServo(args.port, args.baud)  # port kosong = cetak pose saja
    try:
        servo.buka()
    except ImportError:
        log.error("pyserial belum terpasang: pip install pyserial")
        return 1
    except Exception as exc:
        log.error("Gagal membuka port servo %s: %s (cek --list-ports)", args.port, exc)
        return 1

    try:
        pengendali = PengendaliEkspresi(servo, args.threshold, args.smooth)
        return jalankan(args, pengendali.loop)
    finally:
        servo.tutup()  # servo: kirim netral lalu tutup port


if __name__ == "__main__":
    raise SystemExit(main())
