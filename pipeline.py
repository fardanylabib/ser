"""Pipeline: mic -> resample 16 kHz -> Silero VAD -> jendela ucapan -> thread inferensi."""

import argparse
import logging
import queue
import threading
from typing import Any, Callable, Optional

import numpy as np

from audio import PenangkapAudio, buat_resampler
from konfigurasi import SAMPLE_RATE, VAD_FRAME
from model_emosi import PengenalEmosi
from vad import SegmenterVAD

log = logging.getLogger("ser.pipeline")

AKHIR = "akhir"
"""Penanda di antrian segmen: frasa sudah berhenti."""


def kirim_terbaru(antrian: queue.Queue, item: Any) -> None:
    """Masukkan item; kalau antrian penuh, buang yang terlama."""
    while True:
        try:
            antrian.put_nowait(item)
            return
        except queue.Full:
            try:
                antrian.get_nowait()
            except queue.Empty:
                pass


def buka_mic(device: Optional[int], antrian: queue.Queue) -> Optional[PenangkapAudio]:
    """Buka mic; kembalikan None (dan catat error) kalau gagal."""
    try:
        return PenangkapAudio(device, antrian)
    except Exception as exc:
        log.error("Gagal membuka mic: %s", exc)
        log.error("Cek daftar device dengan: python main.py --list-devices")
        return None


def uji_rms(args: argparse.Namespace) -> int:
    """Mode uji: cetak level RMS mic supaya jelas audio benar-benar masuk."""
    antrian_audio: queue.Queue = queue.Queue()
    penangkap = buka_mic(args.device, antrian_audio)
    if penangkap is None:
        return 1

    print(
        f"[rms] device={penangkap.device} samplerate={penangkap.samplerate} Hz"
        "  (Ctrl+C untuk berhenti)",
        flush=True,
    )
    try:
        with penangkap:
            while True:
                try:
                    blok = antrian_audio.get(timeout=0.5)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(blok))))
                bar = "#" * min(50, int(rms * 400))
                print(f"\rrms {rms:.4f} |{bar:<50}|", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[berhenti]", flush=True)
    return 0


def jalankan(
    args: argparse.Namespace,
    loop_fn: Callable[[PengenalEmosi, queue.Queue, threading.Event], None],
) -> int:
    """Jalankan sampai Ctrl+C; `loop_fn` jalan di thread inferensi."""
    antrian_audio: queue.Queue = queue.Queue()
    # Antrian pendek: jendela lama dibuang kalau inferensi tertinggal.
    antrian_segmen: queue.Queue = queue.Queue(maxsize=2)
    berhenti = threading.Event()

    # Buka mic dulu agar error device cepat ketahuan.
    penangkap = buka_mic(args.device, antrian_audio)
    if penangkap is None:
        return 1
    resample = buat_resampler(penangkap.samplerate)

    # VAD
    try:
        from silero_vad import VADIterator, load_silero_vad

        vad = VADIterator(
            load_silero_vad(),
            sampling_rate=SAMPLE_RATE,
            threshold=args.vad_threshold,
            min_silence_duration_ms=400,
        )
    except Exception as exc:
        log.error("Gagal memuat Silero VAD: %s", exc)
        return 1

    # Model emosi
    try:
        pengenal = PengenalEmosi(args.model)
    except Exception as exc:
        log.error("Gagal memuat model emosi: %s", exc)
        log.error("Butuh koneksi internet saat pertama kali (download ke cache HF).")
        return 1

    pekerja = threading.Thread(
        target=loop_fn,
        args=(pengenal, antrian_segmen, berhenti),
        daemon=True,
        name="inferensi",
    )
    pekerja.start()

    segmenter = SegmenterVAD(vad, window_sec=args.window, hop_sec=args.hop)
    sisa = np.empty(0, dtype=np.float32)

    print("[listening...]  (Ctrl+C untuk berhenti)", flush=True)

    try:
        with penangkap:
            while True:
                try:
                    blok = antrian_audio.get(timeout=0.5)
                except queue.Empty:
                    continue

                sisa = np.concatenate([sisa, resample(blok)])

                # Per frame 512 sampel sesuai Silero VAD.
                while len(sisa) >= VAD_FRAME:
                    frame, sisa = sisa[:VAD_FRAME], sisa[VAD_FRAME:]
                    mulai, audio, selesai = segmenter.proses(frame)

                    if mulai:
                        print("\U0001f399  speech detected", flush=True)
                    if audio is not None:
                        kirim_terbaru(antrian_segmen, audio)
                    if selesai:
                        kirim_terbaru(antrian_segmen, AKHIR)
    except KeyboardInterrupt:
        print("\n[berhenti]", flush=True)
    finally:
        berhenti.set()
        # Sentinel agar thread inferensi langsung berhenti.
        kirim_terbaru(antrian_segmen, None)
        pekerja.join(timeout=2.0)
        vad.reset_states()

    return 0
