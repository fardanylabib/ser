"""SER Terminal v3 — wajah GUI beranimasi (tkinter) dari pose main2.py.

Pipeline sama dengan main2.py (mic -> VAD -> wav2vec2 -> skor 5 ekspresi ->
`main2.hitung_pose`). Pose 4-DOF itu tidak dicetak, melainkan digambar jadi
wajah kartun di jendela tkinter (bawaan Python, tanpa dependensi baru) yang
bergerak halus menyusul pose target, plus kedip acak.

Kelas-kelas pendukung dipecah ke modul sendiri:
    constants.py        : konstanta bersama (warna, ukuran kanvas, dof netral, ...)
    kanvas_leher.py      : KanvasLeher — kanvas yang bisa dimiringkan/dianggukkan
    keadaan.py           : Keadaan + singleton KEADAAN (jembatan inferensi -> GUI)
    segmenter_pantau.py  : SegmenterPantau — SegmenterVAD + laporan level suara
    wajah.py             : Wajah — gambar wajah kartun dari pose 4-DOF

Thread:
    utama     : tkinter (wajib di thread utama, terutama di macOS), ~30 fps
    pipeline  : ser.main() — capture + VAD, memutar loop_wajah di thread inferensi

Setelah IDLE_SEC tanpa tebakan baru, wajah pelan-pelan kembali ke netral.

Jalankan:
    python main3.py
    python main3.py --device 2 --verbose
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from typing import Optional

import numpy as np

import main as ser
import main2 as pose
from constants import BOBOT_EKSPRESI, FPS
from keadaan import KEADAAN
from segmenter_pantau import SegmenterPantau
from wajah import Wajah


def loop_wajah(
    pengenal: "ser.PengenalEmosi",
    antrian_segmen: queue.Queue,
    berhenti: threading.Event,
    ambang: float,
    alpha: float = ser.SMOOTH_ALPHA,
) -> None:
    """Sama seperti main2.loop_pose, tapi hasilnya dikirim ke GUI, bukan dicetak."""
    peta = ser.peta_ekspresi(pengenal.labels)
    ema: Optional[np.ndarray] = None

    while not berhenti.is_set():
        try:
            item = antrian_segmen.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:  # sinyal shutdown
            break
        if isinstance(item, str) and item == ser.AKHIR:
            ema = None  # frasa baru dinilai segar
            continue

        try:
            probs = pengenal.probabilitas(item)
        except Exception as exc:  # jangan sampai thread mati gara-gara 1 jendela
            pose.log.error("Inferensi gagal: %s", exc)
            continue

        ema = probs if ema is None else (1.0 - alpha) * ema + alpha * probs

        skor: dict[str, float] = {}
        for i, p in enumerate(ema):
            skor[peta[i]] = skor.get(peta[i], 0.0) + float(p)
        skor = {e: v * BOBOT_EKSPRESI.get(e, 1.0) for e, v in skor.items()}
        total = sum(skor.values()) or 1.0
        skor = {e: v / total for e, v in skor.items()}
        juara = max(skor, key=lambda e: skor[e])
        p4 = pose.hitung_pose(skor, ambang)
        # Log untuk diagnosis: label mentah model (sebelum digabung ke ekspresi)
        # berguna untuk memeriksa urutan label dan penggabungan angry+disgust.
        pose.log.debug(
            "label mentah: %s",
            {pengenal.labels[i]: round(float(p), 2) for i, p in enumerate(ema)},
        )
        pose.log.debug(
            "skor: %s | pose: %s",
            {e: round(s, 2) for e, s in sorted(skor.items(), key=lambda x: -x[1])},
            pose.format_pose(p4),
        )
        # Di bawah ambang wajah sudah memudar ke netral; jangan beri label palsu.
        if skor[juara] < ambang:
            KEADAAN.set(p4, "netral", 0.0)
        else:
            KEADAAN.set(p4, juara, skor[juara])


def main() -> int:
    hasil: list[int] = []

    def jalankan_pipeline() -> None:
        hasil.append(ser.main(loop_wajah))

    root = tk.Tk()
    root.title("SER — wajah")
    root.configure(bg="#1e1b1a")
    wajah = Wajah(root)

    ser.SegmenterVAD = SegmenterPantau  # sisipkan pemantau level suara

    # Pipeline berjalan di thread daemon (mic + model + VAD); GUI di thread utama.
    # Menutup jendela mengakhiri proses, jadi stream mic ikut tertutup oleh OS.
    pipeline = threading.Thread(target=jalankan_pipeline, daemon=True, name="pipeline")
    pipeline.start()

    def putaran() -> None:
        if not pipeline.is_alive():  # mic/model gagal dimuat, atau --list-devices
            root.destroy()
            return
        wajah.tick()
        root.after(int(1000 / FPS), putaran)

    root.after(0, putaran)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return hasil[0] if hasil else 0


if __name__ == "__main__":
    raise SystemExit(main())
