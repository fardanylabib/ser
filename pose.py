"""Tabel pose wajah (7 gerakan -> 11 servo) dan blend skor ekspresi jadi pose.

Skor TIDAK dipakai memilih satu ekspresi pemenang, melainkan mem-blend pose.
Tiap ekspresi punya pose penuh, dan pose akhir = jumlah pose berbobot skor,
dikalikan gerbang keyakinan:

    pose = gate(conf) * sum( skor[e] * POSE_PENUH[e] )

Konsekuensinya output kontinu: ragu antara dua ekspresi menghasilkan wajah
campuran, bukan lompatan; makin yakin modelnya, makin penuh amplitudo pose.
Ini pendekatan interpolasi kontinu ala robot sosial (Kismet, MIT), dengan tabel
pose per-ekspresi yang mengikuti FACS (Facial Action Coding System).

Ada dua lapis:
    7 GERAKAN (DOF)  yang dihitung dari emosi, skala -10..10, 0 = netral
    11 SERVO         tiap gerakan dibagikan ke servo yang sesuai (lihat SERVO)

Gerakan:
    buka mata    -10 merem .. 10 melotot
    angkat alis  -10 mengernyit .. 10 naik
    mulut x      -10 menguncup .. 10 senyum lebar (sudut mulut)
    mulut y        0 mingkem .. 10 menganga
    mata x       -10 melirik kiri .. 10 kanan (arah pandang robot)
    mata y       -10 menunduk .. 10 mendongak
    sudut mata   -10 miring sedih (ujung luar turun) .. 10 miring marah (ujung dalam turun)
"""

from __future__ import annotations

DOF = ("buka mata", "angkat alis", "mulut x", "mulut y", "mata x", "mata y", "sudut mata")
"""Gerakan yang dihitung dari emosi."""

NETRAL: dict[str, float] = {dof: 0.0 for dof in DOF}
"""Pose netral (semua gerakan 0)."""

BATAS: dict[str, tuple[int, int]] = {
    "buka mata": (-10, 10),
    "angkat alis": (-10, 10),
    "mulut x": (-10, 10),
    "mulut y": (0, 10),
    "mata x": (-10, 10),
    "mata y": (-10, 10),
    "sudut mata": (-10, 10),
}

SERVO: tuple[tuple[str, str], ...] = (
    ("kelopak kiri", "buka mata"),
    ("kelopak kanan", "buka mata"),
    ("alis kiri", "angkat alis"),
    ("alis kanan", "angkat alis"),
    ("sudut mulut kiri", "mulut x"),
    ("sudut mulut kanan", "mulut x"),
    ("mulut buka", "mulut y"),
    ("mata kiri x", "mata x"),
    ("mata kanan x", "mata x"),
    ("mata y", "mata y"),
    ("sudut mata", "sudut mata"),
)
"""11 servo, urutannya = nomor channel PCA9685 (servo pertama di channel 0).
Tiap entri: (nama servo, gerakan sumbernya). Kiri/kanan mendapat nilai sama;
arah servo yang terpasang berlawanan dibalik lewat KALIBRASI di servo.py."""

# Pose penuh tiap ekspresi (amplitudo saat model yakin 100%). Angka mengikuti
# kombinasi Action Unit FACS untuk tiap emosi dasar:
#   senyum : AU6+12  -> pipi naik menyipitkan mata, sudut bibir melebar
#   sedih  : AU1+15  -> mata sayu, ujung dalam alis naik, mulut menyempit
#   marah  : AU4+5+23-> alis mengernyit turun, mata melotot, bibir terkatup rapat
#   kaget  : AU1+2+5+26 -> alis & mata terbuka penuh, rahang jatuh, mulut "O"
# `mata y` dan `sudut mata` adalah tebakan (kaget mendongak, sedih menunduk,
# marah menatap tajam dengan sudut mata miring). Ini titik awal yang wajar,
# bukan angka keramat — silakan tuning ke mekanikmu.
#                 buka  alis  mulut mulut mata  mata  sudut
#                 mata        x     y     x     y     mata
POSE_PENUH: dict[str, dict[str, float]] = {
    "netral": dict(zip(DOF, (0, 0, 0, 0, 0, 0, 0))),
    "senyum": dict(zip(DOF, (-3, 1, 8, 3, 0, 1, 0))),
    "sedih": dict(zip(DOF, (-4, 2, -4, 0, 0, -3, -5))),
    "marah": dict(zip(DOF, (4, -8, -5, 1, 0, -1, 6))),
    "kaget": dict(zip(DOF, (8, 8, -2, 7, 0, 3, 0))),
}

GATE_PENUH = 0.85
"""Skor pemenang saat amplitudo pose mencapai 100%. Di antara ambang bawah
(--threshold) dan nilai ini, amplitudo naik linier — wajah "fade in" mengikuti
keyakinan model, tidak langsung full."""


def hitung_pose(skor: dict[str, float], ambang: float) -> dict[str, float]:
    """Blend skor ekspresi jadi satu pose (7 gerakan).

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
    """Susun baris teks, mis. 'buka mata -3; angkat alis 1; mulut x 8; ...'."""
    return "; ".join(f"{dof} {round(pose[dof])}" for dof in DOF)
