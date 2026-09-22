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
from typing import Any, Optional

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
    "sedih": (-5.0, 16.0),  # menunduk lesu
    "marah": (0.0, 9.0),  # dagu masuk, menatap tajam
    "kaget": (0.0, -16.0),  # kepala tersentak ke belakang/atas
}

# Lirikan acak saat tidak ada emosi: kanan, kiri, atas, bawah.
LEHER_LIRIK: list[tuple[float, float]] = [(8.0, 0.0), (-8.0, 0.0), (0.0, -12.0), (0.0, 12.0)]


class KanvasLeher:
    """Pembungkus Canvas yang memiringkan/menganggukkan semua gambar kepala.

    tkinter tidak bisa memutar oval, jadi oval diubah jadi poligon. Miring = putar
    di sekitar dasar leher; angguk = fitur wajah bergeser lebih jauh daripada
    kontur kepala sehingga terkesan menunduk/mendongak.
    """

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self._cos, self._sin, self.angguk = 1.0, 0.0, 0.0

    def atur(self, miring_deg: float, angguk: float) -> None:
        r = math.radians(miring_deg)
        self._cos, self._sin, self.angguk = math.cos(r), math.sin(r), angguk

    def t(self, x: float, y: float, kepala: bool = False) -> tuple[float, float]:
        y += self.angguk * (0.4 if kepala else 1.0)
        dx, dy = x - CX, y - PIVOT_Y
        return CX + dx * self._cos - dy * self._sin, PIVOT_Y + dx * self._sin + dy * self._cos

    def _titik(self, koord: tuple[float, ...], kepala: bool) -> list[float]:
        if len(koord) == 1:  # tkinter juga menerima satu list koordinat
            koord = tuple(koord[0])
        out: list[float] = []
        for i in range(0, len(koord), 2):
            out.extend(self.t(koord[i], koord[i + 1], kepala))
        return out

    def delete(self, *a: Any) -> None:
        self.canvas.delete(*a)

    def create_oval(self, x0: float, y0: float, x1: float, y1: float,
                    kepala: bool = False, **kw: Any) -> int:
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        pts: list[float] = []
        for i in range(36):
            a = 2 * math.pi * i / 36
            pts += [cx + rx * math.cos(a), cy + ry * math.sin(a)]
        return self.canvas.create_polygon(*self._titik(tuple(pts), kepala), **kw)

    def create_line(self, *koord: float, **kw: Any) -> int:
        return self.canvas.create_line(*self._titik(koord, False), **kw)

    def create_polygon(self, *koord: float, **kw: Any) -> int:
        return self.canvas.create_polygon(*self._titik(koord, False), **kw)

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


# --------------------------------------------------------------------------- #
# Menggambar wajah
# --------------------------------------------------------------------------- #


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
        miring = -min(44.0, -mulut_x * 11.0)
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
        # + = sudut bibir naik (senyum); cemberut (mx < 0) dilengkungkan 2x lebih
        # tajam supaya sedih tidak samar dibanding netral.
        lengkung = mx * (2.5 if mx > 0 else 5.0)
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
