"""Tabel pose wajah (7 gerakan -> 11 servo) dan blend skor ekspresi jadi pose."""

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
"""11 servo (urutan = channel PCA9685): (nama servo, gerakan sumbernya)."""

# Pose penuh tiap ekspresi, urutan kolom mengikuti DOF.
POSE_PENUH: dict[str, dict[str, float]] = {
    "netral": dict(zip(DOF, (0, 0, 0, 0, 0, 0, 0))),
    "senyum": dict(zip(DOF, (-3, 1, 8, 3, 0, 1, 0))),
    "sedih": dict(zip(DOF, (-4, 2, -4, 0, 0, -3, -5))),
    "marah": dict(zip(DOF, (4, -8, -5, 1, 0, -1, 6))),
    "kaget": dict(zip(DOF, (8, 8, -2, 7, 0, 3, 0))),
}

GATE_PENUH = 0.85
"""Skor pemenang saat amplitudo pose penuh (di bawahnya naik linier dari ambang)."""


def hitung_pose(skor: dict[str, float], ambang: float) -> dict[str, float]:
    """Blend skor ekspresi jadi satu pose (7 gerakan)."""
    maks = max(skor.values())
    rentang = max(0.05, GATE_PENUH - ambang)  # jaga threshold aneh
    gate = min(1.0, max(0.0, (maks - ambang) / rentang))

    pose = {}
    for dof in DOF:
        nilai = gate * sum(s * POSE_PENUH[e][dof] for e, s in skor.items())
        lo, hi = BATAS[dof]
        pose[dof] = min(float(hi), max(float(lo), nilai))
    return pose


def format_pose(pose: dict[str, float]) -> str:
    """Susun baris teks pose, mis. 'buka mata -3; angkat alis 1; ...'."""
    return "; ".join(f"{dof} {round(pose[dof])}" for dof in DOF)
