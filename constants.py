"""Konstanta bersama untuk GUI wajah (main3.py dan modul pendukungnya)."""

from __future__ import annotations

import main2 as pose

FPS = 30
"""Frame per detik animasi wajah."""

TAU = 0.12
"""Konstanta waktu (detik) gerak wajah menuju pose target. Kecil = lebih sigap."""

IDLE_SEC = 2.5
"""Tanpa tebakan baru selama ini, target pose dikembalikan ke netral."""

WARNA_WAJAH: dict[str, str] = {
    "senyum": "#ffd54a",
    "netral": "#e9e2d0",
    "sedih": "#8ec5f0",
    "marah": "#f07a6a",
    "kaget": "#d9a6f2",
}

BOBOT_EKSPRESI: dict[str, float] = {"sedih": 1.8, "netral": 0.7}
"""Koreksi skor per ekspresi sebelum memilih juara (lalu dinormalkan ulang).

Model dilatih pada akting dan menggabungkan 'calm' + 'neutral' ke netral, jadi
bicara pelan/datar yang sebenarnya sedih hampir selalu jatuh ke netral. Bobot
ini menggeser keputusan ke arah sedih. Heuristik, bukan hasil pengukuran:
kalau terlalu sering salah sedih, turunkan 1.8; kalau masih kurang, naikkan."""

NETRAL = {dof: 0.0 for dof in pose.DOF}

CW, CH = 420, 570
"""Ukuran kanvas."""

CX = CW // 2
TINTA = "#2b2320"

PIVOT_Y = 480
"""Titik putar leher (dasar leher) pada kanvas."""

TAU_LEHER = 0.35
"""Konstanta waktu gerak leher; lebih lambat dari wajah supaya terasa berat/alami."""

# Gerak kepala per ekspresi: (miring derajat, angguk px). Miring + = ke kanan layar,
# angguk + = menunduk, - = mendongak. Dikali keyakinan model.
LEHER_EKSPRESI: dict[str, tuple[float, float]] = {
    "senyum": (6.0, -5.0),  # miring ceria, dagu sedikit naik
    "sedih": (-4.0, 11.0),  # menunduk lesu
    "marah": (0.0, 9.0),  # dagu masuk, menatap tajam
    "kaget": (0.0, -16.0),  # kepala tersentak ke belakang/atas
}

# Lirikan acak saat tidak ada emosi: kanan, kiri, atas, bawah.
LEHER_LIRIK: list[tuple[float, float]] = [(8.0, 0.0), (-8.0, 0.0), (0.0, -12.0), (0.0, 12.0)]
