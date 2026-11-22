"""SegmenterVAD yang juga melaporkan level suara per frame untuk animasi mulut."""

from __future__ import annotations

import numpy as np

import main as ser
from keadaan import KEADAAN


class SegmenterPantau(ser.SegmenterVAD):
    """SegmenterVAD biasa + laporan level suara per frame untuk animasi mulut.

    main.py tidak diubah: `jalankan` mencari nama `SegmenterVAD` saat dipanggil,
    jadi cukup ditimpa dengan subclass ini sebelum pipeline dimulai (lihat main3.main).
    """

    def proses(self, frame: np.ndarray) -> "ser.Peristiwa":
        ev = super().proses(frame)
        rms = float(np.sqrt(np.mean(np.square(frame))))
        KEADAAN.set_suara(self.sedang_bicara, rms)
        return ev
