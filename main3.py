"""SER Terminal v3 — wajah GUI beranimasi (tkinter) dari pose main2.py.

Pipeline sama dengan main2.py (mic -> VAD -> wav2vec2 -> skor 5 ekspresi ->
`main2.hitung_pose`). Pose 4-DOF itu tidak dicetak, melainkan digambar jadi
wajah kartun di jendela tkinter (bawaan Python, tanpa dependensi baru) yang
bergerak halus menyusul pose target, plus kedip acak.

Thread:
    utama     : tkinter (wajib di thread utama, terutama di macOS), ~30 fps
    pipeline  : ser.main() — capture + VAD, memutar loop_wajah di thread inferensi

Setelah IDLE_SEC tanpa tebakan baru, wajah pelan-pelan kembali ke netral.

Jalankan:
    python main3.py
    python main3.py --device 2 --verbose
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
import tkinter as tk
from typing import Optional

import numpy as np

import main as ser
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

NETRAL = {dof: 0.0 for dof in pose.DOF}

CW, CH = 420, 500
"""Ukuran kanvas."""

CX = CW // 2
TINTA = "#2b2320"

# --------------------------------------------------------------------------- #
# Keadaan bersama (inferensi -> GUI)
# --------------------------------------------------------------------------- #


class Keadaan:
    """Target pose dari thread inferensi, dibaca thread GUI (dijaga lock)."""

    def __init__(self) -> None:
        self._kunci = threading.Lock()
        self._target = dict(NETRAL)
        self._label = "netral"
        self._conf = 0.0
        self._waktu = 0.0
        self._bicara = False
        self._rms = 0.0
        self._waktu_suara = 0.0

    def set_suara(self, bicara: bool, rms: float) -> None:
        """Dipanggil tiap frame 32 ms: apakah VAD bilang ada ucapan + level RMS-nya."""
        with self._kunci:
            self._bicara, self._rms = bicara, rms
            self._waktu_suara = time.monotonic()

    def baca_suara(self) -> tuple[bool, float]:
        """Kembalikan (sedang bicara, rms); dianggap diam kalau frame berhenti datang."""
        with self._kunci:
            if time.monotonic() - self._waktu_suara > 0.3:
                return False, 0.0
            return self._bicara, self._rms

    def set(self, target: dict[str, float], label: str, conf: float) -> None:
        with self._kunci:
            self._target, self._label, self._conf = target, label, conf
            self._waktu = time.monotonic()

    def baca(self) -> tuple[dict[str, float], str, float]:
        """Kembalikan (target, label, conf); netral kalau sudah lama diam."""
        with self._kunci:
            if time.monotonic() - self._waktu > IDLE_SEC:
                return dict(NETRAL), "netral", 0.0
            return dict(self._target), self._label, self._conf


KEADAAN = Keadaan()


class SegmenterPantau(ser.SegmenterVAD):
    """SegmenterVAD biasa + laporan level suara per frame untuk animasi mulut.

    main.py tidak diubah: `jalankan` mencari nama `SegmenterVAD` saat dipanggil,
    jadi cukup ditimpa dengan subclass ini sebelum pipeline dimulai (lihat main).
    """

    def proses(self, frame: np.ndarray) -> "ser.Peristiwa":
        ev = super().proses(frame)
        rms = float(np.sqrt(np.mean(np.square(frame))))
        KEADAAN.set_suara(self.sedang_bicara, rms)
        return ev


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


# --------------------------------------------------------------------------- #
# Menggambar wajah
# --------------------------------------------------------------------------- #


def _garis_alis(sisi: int, alis: float, mulut_x: float) -> tuple[float, float, float, float]:
    """Koordinat alis (x0, y0, x1, y1); sisi -1 = kiri, +1 = kanan.

    DOF cuma punya satu angka alis, jadi kemiringan ditebak dari kombinasi:
    mengernyit = ujung dalam turun (marah); naik sedikit + mulut turun = ujung
    dalam naik (sedih); naik banyak = alis tinggi melengkung (kaget).
    """
    y = 125 - alis * 3.5
    miring = 0.0  # + = ujung dalam turun
    if alis <= -3:
        miring = -alis * 3.0
    elif 0 <= alis < 5 and mulut_x < -2:
        miring = -min(14.0, -mulut_x * 3.0)
    luar = CX + sisi * 105
    dalam = CX + sisi * 32
    return luar, y - miring * 0.4, dalam, y + miring * 0.6


class Wajah:
    """Kanvas tkinter + pose sekarang yang bergerak menuju target."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.canvas = tk.Canvas(root, width=CW, height=CH, bg="#1e1b1a", highlightthickness=0)
        self.canvas.pack()
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

        self.gambar(label, conf, kedip)

    def gambar(self, label: str, conf: float, kedip: bool) -> None:
        c = self.canvas
        p = self.sekarang
        c.delete("all")

        # Kepala.
        c.create_oval(50, 40, CW - 50, 420, fill=WARNA_WAJAH.get(label, "#e9e2d0"),
                      outline=TINTA, width=6)

        # Alis.
        for sisi in (-1, 1):
            x0, y0, x1, y1 = _garis_alis(sisi, p["angkat alis"], p["mulut x"])
            c.create_line(x0, y0, x1, y1, width=10, capstyle=tk.ROUND, fill=TINTA)

        # Mata: tinggi mengikuti buka mata (-10 merem .. 10 melotot).
        buka = p["buka mata"]
        tinggi = 5 if kedip else 8 + (buka + 10) / 20 * 52
        for sisi in (-1, 1):
            ex, ey = CX + sisi * 70, 190
            c.create_oval(ex - 30, ey - tinggi / 2, ex + 30, ey + tinggi / 2,
                          fill="white", outline=TINTA, width=4)
            r = min(13.0, tinggi / 2 - 3)
            if r > 2:
                c.create_oval(ex - r, ey - r, ex + r, ey + r, fill=TINTA, outline="")

        # Hidung.
        c.create_line(CX, 235, CX - 8, 262, CX + 8, 262, width=4, fill=TINTA,
                      capstyle=tk.ROUND, joinstyle=tk.ROUND)

        # Mulut.
        # Bukaan bicara ditumpuk di atas pose (mulut senyum tetap senyum saat bicara).
        self.gambar_mulut(p["mulut x"], max(p["mulut y"], 8.5 * self.talk))

        # Keterangan.
        teks = f"{label}  (conf {conf:.2f})" if conf else "mendengarkan..."
        c.create_text(CX, 465, text=teks, fill="#e9e2d0", font=("Helvetica", 20))

    def gambar_mulut(self, mx: float, my: float) -> None:
        c = self.canvas
        y = 320
        hw = 45 + mx * 3.5  # setengah lebar: menguncup .. senyum lebar
        lengkung = mx * 2.5  # + = sudut bibir naik (senyum)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


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
