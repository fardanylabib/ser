"""Kanvas tkinter yang menggambar wajah kartun dari pose 4-DOF."""

from __future__ import annotations

import math
import random
import time
import tkinter as tk

import main2 as pose
from constants import (
    CH,
    CW,
    CX,
    LEHER_EKSPRESI,
    LEHER_LIRIK,
    NETRAL,
    PIVOT_Y,
    TAU,
    TAU_LEHER,
    TINTA,
    WARNA_WAJAH,
)
from kanvas_leher import KanvasLeher
from keadaan import KEADAAN


def _garis_alis(sisi: int, alis: float, mulut_x: float) -> tuple[float, float, float, float]:
    """Koordinat alis (x0, y0, x1, y1); sisi -1 = kiri, +1 = kanan.

    DOF cuma punya satu angka alis, jadi kemiringan ditebak dari kombinasi:
    mengernyit = ujung dalam turun (marah); naik sedikit + mulut turun = ujung
    dalam naik (sedih); naik banyak = alis tinggi melengkung (kaget).
    """
    y = 125 - alis * (3.5 if alis > 0 else 2.5)  # mengernyit: turun lebih sedikit
    miring = 0.0  # + = ujung dalam turun
    if alis <= -2:
        # Marah: ujung dalam (dekat hidung) turun tajam, ujung luar naik.
        miring = min(46.0, -alis * 5.5)
    elif 0 <= alis < 5 and mulut_x < -1:
        # Sedih: ujung luar jatuh ke samping (alis "melorot"), ujung dalam naik.
        miring = -min(30.0, -mulut_x * 8.0)
    luar = CX + sisi * 105
    dalam = CX + sisi * 32
    if miring < 0:  # sedih: dominan ujung luar yang turun, bukan ujung dalam yang naik
        return luar, y - miring * 0.75, dalam, y + miring * 0.25
    return luar, y - miring * 0.5, dalam, y + miring * 0.5


class Wajah:
    """Kanvas tkinter + pose sekarang yang bergerak menuju target."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.canvas = tk.Canvas(root, width=CW, height=CH, bg="#1e1b1a", highlightthickness=0)
        self.canvas.pack()
        self.kv = KanvasLeher(self.canvas)
        self.leher = (0.0, 0.0)  # (miring deg, angguk px) sekarang
        self.lirik = (0.0, 0.0)
        self.lirik_berikut = time.monotonic() + random.uniform(3, 6)
        self.sekarang = dict(NETRAL)
        self.kedip_berikut = time.monotonic() + random.uniform(2, 5)
        self.kedip_sampai = 0.0
        self.terakhir = time.monotonic()
        self.talk = 0.0  # bukaan mulut karena bicara, 0..1
        self.puncak = 0.05  # level suara "keras" yang beradaptasi dengan mic

    def tick(self) -> None:
        t = time.monotonic()
        dt = t - self.terakhir
        self.terakhir = t
        target, label, conf = KEADAAN.baca()

        # Mulut berkecap: bukaan mengikuti level suara, dinormalisasi ke puncak
        # terbaru supaya tidak bergantung pada gain mic. Naik cepat, turun agak
        # pelan — kira-kira laju suku kata (4-5 Hz) jadi terlihat seperti bicara.
        bicara, rms = KEADAAN.baca_suara()
        self.puncak = max(0.02, rms, self.puncak * math.exp(-dt / 3.0))
        tujuan = (min(1.0, rms / self.puncak) ** 0.8) if bicara else 0.0
        tk_ = 0.03 if tujuan > self.talk else 0.09
        self.talk += (tujuan - self.talk) * (1.0 - math.exp(-dt / tk_))

        # Geser eksponensial: stabil berapa pun dt, tanpa overshoot.
        k = 1.0 - math.exp(-dt / TAU)
        for dof in pose.DOF:
            self.sekarang[dof] += (target[dof] - self.sekarang[dof]) * k

        # Kedip acak, kecuali mata sedang melotot (kaget tidak berkedip).
        if t >= self.kedip_berikut:
            self.kedip_sampai = t + 0.14
            self.kedip_berikut = t + random.uniform(2, 5)
        kedip = t < self.kedip_sampai and self.sekarang["buka mata"] <= 4

        self.gerak_leher(t, dt, label, conf)
        self.gambar(label, conf, kedip)

    def gerak_leher(self, t: float, dt: float, label: str, conf: float) -> None:
        """Hitung miring/angguk kepala: pose emosi, lirikan acak, anggukan bicara."""
        if label in LEHER_EKSPRESI and conf:
            amp = min(1.0, conf / pose.GATE_PENUH)
            tm, ta = (v * amp for v in LEHER_EKSPRESI[label])
        else:
            if t >= self.lirik_berikut:  # sesekali melirik (kanan/kiri/atas/bawah) lalu kembali
                self.lirik = random.choice(LEHER_LIRIK) if self.lirik == (0.0, 0.0) else (0.0, 0.0)
                self.lirik_berikut = t + random.uniform(1.2, 2.0 if self.lirik != (0.0, 0.0) else 6.0)
            tm, ta = self.lirik
        # Goyang halus supaya tidak kaku + angguk kecil mengikuti bicara.
        tm += 1.5 * math.sin(t * 0.7)
        ta += 1.5 * math.sin(t * 0.5 + 1.0) + 2.5 * self.talk * math.sin(t * 8.0)
        k = 1.0 - math.exp(-dt / TAU_LEHER)
        self.leher = (self.leher[0] + (tm - self.leher[0]) * k,
                      self.leher[1] + (ta - self.leher[1]) * k)
        self.kv.atur(*self.leher)

    def gambar(self, label: str, conf: float, kedip: bool) -> None:
        c = self.kv
        p = self.sekarang
        c.delete("all")

        # Bahu + leher: dasar tetap di tempat, ujung atas ikut kepala.
        self.canvas.create_oval(CX - 100, PIVOT_Y - 12, CX + 100, PIVOT_Y + 50,
                                fill="#3a3230", outline=TINTA, width=4)
        (ax, ay), (bx, by) = c.t(CX - 30, 400, True), c.t(CX + 30, 400, True)
        self.canvas.create_polygon(ax, ay, bx, by, CX + 30, PIVOT_Y, CX - 30, PIVOT_Y,
                                   fill="#7a6b5f", outline=TINTA, width=4)

        # Kepala.
        warna = WARNA_WAJAH.get(label, "#e9e2d0")
        c.create_oval(50, 40, CW - 50, 420, kepala=True, fill=warna, outline=TINTA, width=6)

        # Mata: -10 merem .. 0 normal .. 10 melotot. Pemetaan tidak linear supaya
        # sipit (< 0) benar-benar menyipit dan melotot (> 0) benar-benar membelalak.
        u = p["buka mata"] / 10.0
        if u < 0:
            tinggi = 32 * (1 + u) ** 1.6  # -3 -> ~18 px, -4 -> ~14 px, -10 -> 0
        else:
            tinggi = 32 + 72 * u  # 4 -> ~61 px, 8 -> ~90 px
        tinggi = max(3.0, tinggi)
        if kedip:
            tinggi = 3.0
        lebar = 30 + max(0.0, u) * 8  # melotot: mata ikut melebar
        for sisi in (-1, 1):
            ex, ey = CX + sisi * 70, 190
            c.create_oval(ex - lebar, ey - tinggi / 2, ex + lebar, ey + tinggi / 2,
                          fill="white", outline=TINTA, width=4)
            # Pupil mengecil saat melotot; saat sipit terpotong kelopak.
            r = min(13.0 + max(0.0, u) * 3, tinggi / 2 - 3)
            if r > 2:
                c.create_oval(ex - r, ey - r, ex + r, ey + r, fill=TINTA, outline="")
            # Marah: kelopak atas miring (ujung dalam turun) memotong mata jadi
            # tatapan tajam. Tanpa ini mata marah yang terbuka mirip kaget.
            garang = min(1.0, max(0.0, (-p["angkat alis"] - 2) / 6.0))
            if garang > 0 and not kedip:
                atas = ey - tinggi / 2
                miring = min(tinggi * 0.6, 26 * garang)
                luar, dalam = ex + sisi * (lebar + 3), ex - sisi * (lebar + 3)
                c.create_polygon(dalam, atas - 8, luar, atas - 8, luar, atas + 3,
                                 dalam, atas + 3 + miring, fill=warna, outline="")
                c.create_line(luar, atas + 3, dalam, atas + 3 + miring,
                              width=5, fill=TINTA, capstyle=tk.ROUND)

        # Sedih: air mata di bawah mata, muncul makin jelas seiring keyakinan.
        sedih = min(1.0, max(0.0, -p["mulut x"] - 1) / 3.0) if 0 <= p["angkat alis"] < 5 else 0.0
        if sedih > 0.5 and not kedip:
            for sisi in (-1, 1):
                tx, ty = CX + sisi * 82, 190 + tinggi / 2 + 12 + 6 * sedih
                c.create_polygon(tx, ty - 12, tx - 6, ty + 2, tx, ty + 9, tx + 6, ty + 2,
                                 smooth=True, fill="#7ec8ff", outline=TINTA, width=2)

        # Alis (digambar setelah mata supaya tidak tertutup kelopak marah).
        for sisi in (-1, 1):
            x0, y0, x1, y1 = _garis_alis(sisi, p["angkat alis"], p["mulut x"])
            c.create_line(x0, y0, x1, y1, width=10, capstyle=tk.ROUND, fill=TINTA)

        # Hidung.
        c.create_line(CX, 235, CX - 8, 262, CX + 8, 262, width=4, fill=TINTA,
                      capstyle=tk.ROUND, joinstyle=tk.ROUND)

        # Mulut.
        # Bukaan bicara ditumpuk di atas pose (mulut senyum tetap senyum saat bicara).
        self.gambar_mulut(p["mulut x"], max(p["mulut y"], 8.5 * self.talk))

        # Keterangan.
        teks = f"{label}  (conf {conf:.2f})" if conf else "mendengarkan..."
        self.canvas.create_text(CX, 545, text=teks, fill="#e9e2d0", font=("Helvetica", 20))

    def gambar_mulut(self, mx: float, my: float) -> None:
        c = self.kv
        y = 320
        hw = 45 + mx * (3.5 if mx > 0 else 1.5)  # setengah lebar: senyum melebar, cemberut tetap lebar
        # + = sudut bibir naik (senyum); cemberut (mx < 0) dilengkungkan sedikit
        # lebih tajam supaya sedih tidak samar dibanding netral.
        lengkung = mx * (2.5 if mx > 0 else 3.5)
        buka = my * 7  # tinggi bukaan (px)
        hw = max(28.0, hw - my * 2.0) if my > 1 else hw  # menganga -> lebih bulat
        if buka < 2.5:
            # Mingkem: satu garis lengkung.
            c.create_line(CX - hw, y - lengkung, CX, y + lengkung, CX + hw, y - lengkung,
                          smooth=True, width=8, fill=TINTA, capstyle=tk.ROUND)
            return
        # Bibir atas = lengkung yang sama dengan mingkem; bibir bawah turun sejauh
        # `buka` dan sedikit menyempit. Bentuknya menyambung mulus dari garis di
        # atas, jadi mulut berkecap tidak berkedip antara dua gaya gambar.
        titik = [
            CX - hw, y - lengkung,
            CX, y + lengkung,
            CX + hw, y - lengkung,
            CX + hw * 0.8, y - lengkung + buka * 0.8,
            CX, y + lengkung + buka,
            CX - hw * 0.8, y - lengkung + buka * 0.8,
        ]
        c.create_polygon(titik, smooth=True, fill="#5a1f1f", outline=TINTA, width=6)
