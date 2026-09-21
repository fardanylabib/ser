"""SER Terminal v2 — output derajat pose wajah (untuk servo), bukan emoji.

Pipeline sama persis dengan main.py sampai tahap skor 5 ekspresi. Bedanya di
langkah terakhir: skor TIDAK dipakai memilih satu ekspresi pemenang, melainkan
mem-blend pose. Tiap ekspresi punya pose penuh (4 derajat kebebasan), dan pose
akhir = jumlah pose berbobot skor, dikalikan gerbang keyakinan:

    pose = gate(conf) * sum( skor[e] * POSE_PENUH[e] )

Konsekuensinya output kontinu: ragu antara dua ekspresi menghasilkan wajah
campuran, bukan lompatan; makin yakin modelnya, makin penuh amplitudo pose.
Ini pendekatan interpolasi kontinu ala robot sosial (Kismet, MIT), dengan tabel
pose per-ekspresi yang mengikuti FACS (Facial Action Coding System).

Derajat kebebasan (skala -10..10, 0 = netral):
    buka mata   -10 full merem .. 10 full melotot
    angkat alis -10 mengernyit .. 10 naik full
    mulut x     -10 menguncup  .. 10 senyum lebar
    mulut y       0 mingkem    .. 10 menganga lebar

Satu baris pose dicetak ke stdout tiap jendela inferensi (~tiap 0.3 dtk selama
ada ucapan), contoh:

    buka mata -3; angkat alis 1; mulut x 8; mulut y 3

Baris lain di stdout hanya banner dan "speech detected" — parser di sisi servo
cukup ambil baris yang diawali "buka mata". Log internal ke stderr.

Jalankan:
    python main2.py
    python main2.py --verbose        # + skor ekspresi mentah di stderr
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np

import main as ser

log = logging.getLogger("ser.pose")

# --------------------------------------------------------------------------- #
# Tabel pose
# --------------------------------------------------------------------------- #

DOF = ("buka mata", "angkat alis", "mulut x", "mulut y")
"""Urutan derajat kebebasan pada baris output."""

BATAS: dict[str, tuple[int, int]] = {
    "buka mata": (-10, 10),
    "angkat alis": (-10, 10),
    "mulut x": (-10, 10),
    "mulut y": (0, 10),
}

# Pose penuh tiap ekspresi (amplitudo saat model yakin 100%). Angka mengikuti
# kombinasi Action Unit FACS untuk tiap emosi dasar, diterjemahkan ke 4 DOF:
#   senyum : AU6+12  -> pipi naik menyipitkan mata, sudut bibir melebar
#   sedih  : AU1+15  -> mata sayu, ujung dalam alis naik, mulut menyempit
#   marah  : AU4+5+23-> alis mengernyit turun, mata melotot, bibir terkatup rapat
#   kaget  : AU1+2+5+26 -> alis & mata terbuka penuh, rahang jatuh, mulut "O"
# Ini titik awal yang wajar, bukan angka keramat — silakan tuning ke mekanikmu.
POSE_PENUH: dict[str, dict[str, float]] = {
    "netral": {"buka mata": 0, "angkat alis": 0, "mulut x": 0, "mulut y": 0},
    "senyum": {"buka mata": -3, "angkat alis": 1, "mulut x": 8, "mulut y": 3},
    "sedih": {"buka mata": -4, "angkat alis": 2, "mulut x": -4, "mulut y": 0},
    "marah": {"buka mata": 4, "angkat alis": -8, "mulut x": -5, "mulut y": 1},
    "kaget": {"buka mata": 8, "angkat alis": 8, "mulut x": -2, "mulut y": 7},
}

GATE_PENUH = 0.85
"""Skor pemenang saat amplitudo pose mencapai 100%. Di antara ambang bawah
(--threshold) dan nilai ini, amplitudo naik linier — wajah "fade in" mengikuti
keyakinan model, tidak langsung full."""


# --------------------------------------------------------------------------- #
# Perhitungan pose
# --------------------------------------------------------------------------- #


def hitung_pose(skor: dict[str, float], ambang: float) -> dict[str, float]:
    """Blend skor ekspresi jadi satu pose 4-DOF.

    Bobot blend = distribusi skor itu sendiri (jumlahnya 1), jadi keraguan
    antara dua ekspresi menghasilkan pose campuran. `gate` menekan amplitudo
    ke netral saat model bingung (skor tertinggi rendah = distribusi datar).
    """
    maks = max(skor.values())
    rentang = max(0.05, GATE_PENUH - ambang)  # jaga-jaga --threshold aneh
    gate = min(1.0, max(0.0, (maks - ambang) / rentang))

    pose = {}
    for dof in DOF:
        nilai = gate * sum(s * POSE_PENUH[e][dof] for e, s in skor.items())
        lo, hi = BATAS[dof]
        pose[dof] = min(float(hi), max(float(lo), nilai))
    return pose


def format_pose(pose: dict[str, float]) -> str:
    """Susun baris output, mis. 'buka mata -3; angkat alis 1; mulut x 8; mulut y 3'."""
    return "; ".join(f"{dof} {round(pose[dof])}" for dof in DOF)


# --------------------------------------------------------------------------- #
# Thread inferensi (menggantikan main.loop_inferensi)
# --------------------------------------------------------------------------- #


def loop_pose(
    pengenal: "ser.PengenalEmosi",
    antrian_segmen: queue.Queue,
    berhenti: threading.Event,
    ambang: float,
    alpha: float = ser.SMOOTH_ALPHA,
) -> None:
    """Konsumsi jendela audio, cetak satu baris pose per jendela.

    Tidak ada histeresis di sini — output kontinu tidak punya "lompatan kelas"
    yang perlu diredam; EMA + gate sudah membuat pose bergerak halus. Saat
    hening, pose terakhir dibiarkan terpasang (tidak ada baris baru); kapan
    wajah kembali ke netral adalah keputusan sisi penggerak servo.
    """
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
            ema = None  # frasa baru dinilai segar; pose terakhir tetap terpasang
            continue

        try:
            probs = pengenal.probabilitas(item)
        except Exception as exc:  # jangan sampai thread mati gara-gara 1 jendela
            log.error("Inferensi gagal: %s", exc)
            continue

        ema = probs if ema is None else (1.0 - alpha) * ema + alpha * probs

        skor: dict[str, float] = {}
        for i, p in enumerate(ema):
            skor[peta[i]] = skor.get(peta[i], 0.0) + float(p)
        log.debug(
            "skor: %s",
            {e: round(s, 2) for e, s in sorted(skor.items(), key=lambda x: -x[1])},
        )

        print(format_pose(hitung_pose(skor, ambang)), flush=True)


if __name__ == "__main__":
    raise SystemExit(ser.main(loop_pose))
