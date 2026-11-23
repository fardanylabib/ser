"""Tabel pose wajah (4 derajat kebebasan) dan perhitungan blend dari skor ekspresi.

Skor TIDAK dipakai memilih satu ekspresi pemenang, melainkan mem-blend pose.
Tiap ekspresi punya pose penuh, dan pose akhir = jumlah pose berbobot skor,
dikalikan gerbang keyakinan:

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
"""

from __future__ import annotations

DOF = ("buka mata", "angkat alis", "mulut x", "mulut y")
"""Urutan derajat kebebasan (juga urutan di protokol UART ke ESP32)."""

NETRAL: dict[str, float] = {dof: 0.0 for dof in DOF}
"""Pose netral (semua DOF 0)."""

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
    """Susun baris teks, mis. 'buka mata -3; angkat alis 1; mulut x 8; mulut y 3'."""
    return "; ".join(f"{dof} {round(pose[dof])}" for dof in DOF)
