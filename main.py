"""SER Terminal — dengar ucapan dari mic, log ekspresi wajah untuk robot.

Emosi ucapan dideteksi dengan wav2vec2 SER, lalu diterjemahkan ke 5 ekspresi
wajah (senyum/netral/sedih/marah/kaget) yang nantinya dipetakan ke pose servo.
Program ini hanya me-log ekspresi; penggerak aktuatornya di luar lingkup.

Alur singkat:
    mic -> callback sounddevice -> antrian audio
        -> resample ke 16 kHz -> Silero VAD (ada ucapan / tidak)
        -> jendela geser 1.5 dtk tiap 0.3 dtk -> thread inferensi
        -> wav2vec2 SER -> agregasi label ke ekspresi -> EMA + histeresis
        -> print ekspresi wajah + confidence

Jalankan:
    python main.py
    python main.py --list-devices
    python main.py --device 2 --rms
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from collections import deque
from typing import Any, Callable, Deque, NamedTuple, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16_000
"""Sample rate wajib untuk Silero VAD maupun model wav2vec2."""

VAD_FRAME = 512
"""Silero VAD pada 16 kHz hanya menerima frame tepat 512 sampel (32 ms)."""

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
"""Model SER siap-pakai (dilatih di RAVDESS, 7 kelas — tanpa 'neutral')."""

CONF_THRESHOLD = 0.40
"""Ambang confidence minimal agar label dicetak."""

MIN_SPEECH_SEC = 0.4
"""Audio minimal sebelum model boleh menebak (di bawah ini terlalu pendek)."""

WINDOW_SEC = 1.5
"""Panjang jendela geser: emosi dinilai dari 1.5 detik ucapan terakhir."""

HOP_SEC = 0.3
"""Jarak antar-inferensi. Tebakan diperbarui tiap 0.3 detik selama bicara."""

PREROLL_SEC = 0.3
"""Audio sebelum VAD memicu 'start', supaya awal kata tidak terpotong."""

SMOOTH_ALPHA = 0.4
"""Bobot tebakan baru saat smoothing. 1.0 = tanpa smoothing, kecil = lebih kalem.

Diukur pada rekaman mic 15 detik: tanpa smoothing label berganti 7x dari 27
tebakan, di 0.4 jadi 5x, di 0.2 jadi 3x. Nilai kecil makin kalem tapi makin
lambat menyusul perubahan emosi yang sungguhan, jadi 0.4 diambil sebagai
kompromi — masih terasa realtime.
"""

GANTI_YAKIN = 0.60
"""Skor minimal agar ekspresi wajah boleh berganti SEKETIKA. Di bawah ini,
ekspresi baru harus menang dua jendela berturut-turut (~0.6 dtk) dulu —
histeresis sederhana supaya wajah robot tidak gemetar ganti pose."""

# Ekspresi wajah robot. Sengaja sedikit dan tegas: tiap ekspresi nantinya jadi
# satu preset pose servo. Label emosi model yang mirip digabung ke ekspresi yang
# sama — kategori kasar begini juga lebih stabil daripada 7-8 label halus.
EKSPRESI_EMOJI: dict[str, str] = {
    "senyum": "\U0001f642",
    "netral": "\U0001f610",
    "sedih": "\U0001f641",
    "marah": "\U0001f620",
    "kaget": "\U0001f632",
}

# Peta label emosi model -> ekspresi wajah. Mencakup nama/singkatan yang dipakai
# model alternatif (mis. superb/wav2vec2-base-superb-er). Probabilitas label yang
# menunjuk ekspresi sama akan DIJUMLAHKAN, jadi keraguan model antara mis.
# 'fearful' dan 'surprised' tidak membuat wajah robot ikut ragu.
LABEL_KE_EKSPRESI: dict[str, str] = {
    "happy": "senyum",
    "happiness": "senyum",
    "hap": "senyum",
    "calm": "netral",
    "neutral": "netral",
    "neu": "netral",
    "sad": "sedih",
    "sadness": "sedih",
    "angry": "marah",
    "ang": "marah",
    "disgust": "marah",
    "fearful": "kaget",
    "fear": "kaget",
    "surprised": "kaget",
    "surprise": "kaget",
}

# Urutan label RAVDESS, dipakai kalau config model hanya berisi LABEL_0..LABEL_7.
RAVDESS_ORDER = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

log = logging.getLogger("ser")


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #


def peta_ekspresi(labels: list[str]) -> list[str]:
    """Bangun peta index-label -> nama ekspresi untuk satu model.

    Label yang tidak dikenal dipetakan ke 'netral' (wajah aman untuk robot)
    sambil dicatat di log, supaya model custom apa pun tetap jalan.
    """
    peta = []
    for label in labels:
        eks = LABEL_KE_EKSPRESI.get(label.lower().strip())
        if eks is None:
            log.warning("Label '%s' tidak dikenal, dipetakan ke ekspresi netral.", label)
            eks = "netral"
        peta.append(eks)
    return peta


def rapikan_nama_label(config: Any) -> list[str]:
    """Ambil daftar label dari config model, dengan fallback urutan RAVDESS."""
    id2label = getattr(config, "id2label", None) or {}
    labels = [str(id2label.get(i, f"LABEL_{i}")) for i in range(len(id2label))]

    # Beberapa checkpoint tidak menyimpan nama label asli, hanya LABEL_n.
    if labels and all(nama.upper().startswith("LABEL_") for nama in labels):
        if len(labels) == len(RAVDESS_ORDER):
            log.warning("Config model tanpa nama label, memakai urutan RAVDESS.")
            return list(RAVDESS_ORDER)
    return labels


# --------------------------------------------------------------------------- #
# Penangkapan audio
# --------------------------------------------------------------------------- #


class PenangkapAudio:
    """Buka input stream sounddevice dan lempar blok audio ke antrian.

    Callback sengaja dibuat sangat ringan (hanya copy + put) agar tidak pernah
    mem-block PortAudio. Resample dan VAD dikerjakan di thread lain.
    """

    def __init__(self, device: Optional[int], antrian: queue.Queue) -> None:
        import sounddevice as sd

        self._sd = sd
        self._antrian = antrian
        self.device = device
        self.samplerate = self._pilih_samplerate(device)
        self.stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            blocksize=int(self.samplerate * 0.032),
            callback=self._callback,
        )

    def _pilih_samplerate(self, device: Optional[int]) -> int:
        """Pakai 16 kHz kalau device sanggup; kalau tidak, pakai rate bawaannya."""
        try:
            self._sd.check_input_settings(
                device=device, channels=1, samplerate=SAMPLE_RATE, dtype="float32"
            )
            return SAMPLE_RATE
        except Exception:
            info = self._sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
            log.info("Device tidak mendukung 16 kHz, memakai %d Hz + resample.", rate)
            return rate

    def _callback(self, indata: np.ndarray, frames: int, waktu: Any, status: Any) -> None:
        if status:
            log.debug("Status stream: %s", status)
        self._antrian.put(indata[:, 0].copy())

    def __enter__(self) -> "PenangkapAudio":
        self.stream.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stream.stop()
        self.stream.close()


# --------------------------------------------------------------------------- #
# Segmentasi ucapan (VAD)
# --------------------------------------------------------------------------- #


class Peristiwa(NamedTuple):
    """Hasil satu frame: apa yang baru terjadi pada aliran ucapan."""

    mulai: bool
    """True tepat saat VAD mendeteksi awal ucapan."""

    audio: Optional[np.ndarray]
    """Jendela audio siap-inferensi, atau None kalau belum waktunya menebak."""

    selesai: bool
    """True tepat saat ucapan berhenti (frasa ditutup)."""


class SegmenterVAD:
    """Ubah aliran frame 512 sampel jadi jendela-jendela audio siap-inferensi.

    Berbeda dari pendekatan "tunggu frasa selesai": selama VAD bilang masih ada
    ucapan, tiap `hop_sec` kelas ini mengeluarkan `window_sec` detik audio
    TERAKHIR. Jadi model menebak berulang kali sambil orangnya masih bicara,
    bukan sekali di akhir kalimat.

    Buffer sengaja dipangkas sepanjang jendela saja — kita hanya peduli emosi
    yang sedang berlangsung, dan itu juga yang menjaga penggunaan memori tetap
    datar berapa lama pun orang bicara.

    Dipisah dari loop mic supaya logikanya bisa diuji dengan file audio biasa.
    """

    def __init__(
        self,
        vad: Any,
        min_sec: float = MIN_SPEECH_SEC,
        window_sec: float = WINDOW_SEC,
        hop_sec: float = HOP_SEC,
        preroll_sec: float = PREROLL_SEC,
    ) -> None:
        self.vad = vad
        self.min_sampel = int(min_sec * SAMPLE_RATE)
        self.maks_sampel = int(window_sec * SAMPLE_RATE)
        self.hop_sampel = int(hop_sec * SAMPLE_RATE)
        self.preroll: Deque[np.ndarray] = deque(
            maxlen=max(1, int(preroll_sec * SAMPLE_RATE / VAD_FRAME))
        )
        self.buffer: Deque[np.ndarray] = deque()
        self.panjang = 0
        self.sejak_tebak = 0
        self.sedang_bicara = False

    def proses(self, frame: np.ndarray) -> Peristiwa:
        """Masukkan satu frame 512 sampel, kembalikan peristiwa yang timbul."""
        mulai = False
        selesai = False
        audio: Optional[np.ndarray] = None

        if self.sedang_bicara:
            self._tambah(frame)
        else:
            self.preroll.append(frame)

        hasil = self.vad(frame.copy(), return_seconds=False)

        if hasil and "start" in hasil and not self.sedang_bicara:
            self.sedang_bicara = True
            # Sertakan pre-roll agar awal kata tidak terpotong.
            self.buffer.clear()
            self.panjang = 0
            self.sejak_tebak = 0
            for f in self.preroll:
                self._tambah(f)
            self.preroll.clear()
            mulai = True

        elif hasil and "end" in hasil and self.sedang_bicara:
            # Tebakan terakhir untuk frasa ini, lalu tutup.
            audio = self._jendela()
            self.sedang_bicara = False
            self.buffer.clear()
            self.panjang = 0
            selesai = True

        elif self.sedang_bicara and self.sejak_tebak >= self.hop_sampel:
            audio = self._jendela()
            self.sejak_tebak = 0

        return Peristiwa(mulai, audio, selesai)

    def _tambah(self, frame: np.ndarray) -> None:
        """Masukkan frame ke buffer, buang yang sudah keluar dari jendela."""
        self.buffer.append(frame)
        self.panjang += len(frame)
        self.sejak_tebak += len(frame)
        while self.panjang - len(self.buffer[0]) >= self.maks_sampel:
            self.panjang -= len(self.buffer.popleft())

    def _jendela(self) -> Optional[np.ndarray]:
        """Salinan isi buffer, atau None kalau audio belum cukup panjang."""
        if self.panjang < self.min_sampel:
            return None  # terlalu pendek, model belum punya bahan
        return np.concatenate(self.buffer)

    def selesaikan(self) -> Optional[np.ndarray]:
        """Tutup ucapan yang masih menggantung (dipakai saat aliran audio habis)."""
        if not self.sedang_bicara:
            return None
        self.sedang_bicara = False
        return self._jendela()


# --------------------------------------------------------------------------- #
# Inferensi emosi
# --------------------------------------------------------------------------- #


class PengenalEmosi:
    """Bungkus model wav2vec2 SER dari HuggingFace. Di-load sekali di awal."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Memuat model %s (device: %s)...", model_id, self.device)

        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        except OSError as exc:
            # Jaringan bermasalah tapi model mungkin sudah ada di cache HF.
            log.warning("Gagal menghubungi HuggingFace (%s), coba pakai cache...", exc)
            self.extractor = AutoFeatureExtractor.from_pretrained(
                model_id, local_files_only=True
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id, local_files_only=True
            )

        self.model.to(self.device).eval()
        self.labels = rapikan_nama_label(self.model.config)
        log.info("Model siap. Label: %s", ", ".join(self.labels))

    def probabilitas(self, audio: np.ndarray) -> np.ndarray:
        """Kembalikan vektor probabilitas semua kelas untuk audio 16 kHz mono."""
        torch = self._torch
        inputs = self.extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def prediksi(self, audio: np.ndarray) -> tuple[str, float]:
        """Kembalikan (label, confidence) untuk satu segmen audio 16 kHz mono."""
        probs = self.probabilitas(audio)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


# --------------------------------------------------------------------------- #
# Thread inferensi
# --------------------------------------------------------------------------- #


AKHIR = "akhir"
"""Penanda di antrian: frasa sudah berhenti, kunci baris terakhir."""


def kirim_terbaru(antrian: queue.Queue, item: Any) -> None:
    """Masukkan item; kalau antrian penuh, buang yang terlama.

    Untuk realtime, tebakan basi tidak ada gunanya — lebih baik dibuang daripada
    membuat output tertinggal makin jauh di belakang suara yang sedang berjalan.
    """
    while True:
        try:
            antrian.put_nowait(item)
            return
        except queue.Full:
            try:
                antrian.get_nowait()
            except queue.Empty:
                pass


def loop_inferensi(
    pengenal: PengenalEmosi,
    antrian_segmen: queue.Queue,
    berhenti: threading.Event,
    ambang: float,
    alpha: float = SMOOTH_ALPHA,
) -> None:
    """Konsumsi jendela audio, tebak emosi, terjemahkan jadi ekspresi wajah.

    Yang dijaga stabil di sini adalah EKSPRESI, bukan label emosi mentah:

    1. Probabilitas label yang menunjuk ekspresi sama dijumlahkan (keraguan
       model antara mis. 'fearful'/'surprised' tidak menggoyang wajah).
    2. Skor di-EMA-kan antar-jendela (`alpha`, 1.0 = tanpa smoothing).
    3. Histeresis: ekspresi berganti hanya kalau pemenang baru menang telak
       (>= GANTI_YAKIN) atau menang dua jendela berturut-turut.

    Ekspresi BERTAHAN melewati jeda hening — wajah robot tetap di pose
    terakhir sampai ada ucapan baru yang menggesernya. Tiap periode ekspresi
    dikunci sebagai satu baris riwayat; baris terakhir diperbarui live.
    """
    peta = peta_ekspresi(pengenal.labels)
    tampil: Optional[str] = None  # ekspresi yang sedang "terpasang" di wajah
    kandidat: Optional[str] = None  # calon pengganti, menunggu menang sekali lagi
    ema: Optional[np.ndarray] = None
    terkunci = True  # baris terminal saat ini sudah ditutup newline?

    while not berhenti.is_set():
        try:
            item = antrian_segmen.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:  # sinyal shutdown
            if not terkunci:
                print(flush=True)
            break

        if isinstance(item, str) and item == AKHIR:
            if not terkunci:
                print(flush=True)  # kunci baris terakhir frasa ini
                terkunci = True
            ema = None  # frasa baru dinilai segar, jangan terbawa audio lama
            kandidat = None
            continue

        try:
            probs = pengenal.probabilitas(item)
        except Exception as exc:  # jangan sampai thread mati gara-gara 1 jendela
            log.error("Inferensi gagal: %s", exc)
            continue

        ema = probs if ema is None else (1.0 - alpha) * ema + alpha * probs

        # Agregasi label -> skor per ekspresi.
        skor: dict[str, float] = {}
        for i, p in enumerate(ema):
            skor[peta[i]] = skor.get(peta[i], 0.0) + float(p)
        juara = max(skor, key=lambda e: skor[e])
        conf = skor[juara]
        log.debug(
            "skor ekspresi: %s",
            {e: round(s, 2) for e, s in sorted(skor.items(), key=lambda x: -x[1])},
        )

        # Kurang yakin? Pertahankan wajah yang sudah terpasang.
        if conf < ambang and tampil is not None:
            juara = tampil
        elif conf < ambang:
            print(f"\r\U0001f937  (belum yakin: {conf:.2f})      ", end="", flush=True)
            terkunci = False
            continue

        # Histeresis pergantian ekspresi.
        if juara != tampil:
            if tampil is None or juara == kandidat or conf >= GANTI_YAKIN:
                if not terkunci:
                    print(flush=True)  # kunci periode ekspresi sebelumnya
                tampil, kandidat = juara, None
            else:
                kandidat = juara  # baru sekali menang; tunggu jendela berikut
        else:
            kandidat = None

        emoji = EKSPRESI_EMOJI[tampil]
        print(
            f"\r{emoji}  {tampil:<8} (conf {skor.get(tampil, 0.0):.2f})     ",
            end="",
            flush=True,
        )
        terkunci = False


# --------------------------------------------------------------------------- #
# Loop utama: capture -> resample -> VAD -> segmen
# --------------------------------------------------------------------------- #


def buat_resampler(rate_sumber: int):
    """Kembalikan fungsi resample ke 16 kHz (identitas kalau rate sudah pas)."""
    if rate_sumber == SAMPLE_RATE:
        return lambda blok: blok

    import torch
    import torchaudio.functional as AF

    def resample(blok: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(blok).unsqueeze(0)
        keluar = AF.resample(tensor, rate_sumber, SAMPLE_RATE)
        return keluar.squeeze(0).numpy()

    return resample


def jalankan(args: argparse.Namespace, loop_fn: Optional[Callable] = None) -> int:
    """Jalankan aplikasi sampai Ctrl+C.

    `loop_fn` adalah fungsi thread inferensi; default `loop_inferensi` (output
    emoji ekspresi). main2.py menyuntikkan loop lain di sini untuk output pose
    servo, tanpa menduplikasi pipeline capture/VAD.
    """
    if loop_fn is None:
        loop_fn = loop_inferensi
    antrian_audio: queue.Queue = queue.Queue()
    # Antrian sengaja pendek: kalau inferensi tertinggal, lebih baik jendela lama
    # dibuang (lihat `kirim_terbaru`) daripada output makin telat dari suaranya.
    antrian_segmen: queue.Queue = queue.Queue(maxsize=2)
    berhenti = threading.Event()

    # --- buka mic dulu, biar error device ketahuan sebelum download model ---
    try:
        penangkap = PenangkapAudio(args.device, antrian_audio)
    except Exception as exc:
        log.error("Gagal membuka mic: %s", exc)
        log.error("Cek daftar device dengan: python main.py --list-devices")
        return 1

    resample = buat_resampler(penangkap.samplerate)

    # --- mode debug: cukup cetak level RMS, tanpa VAD/model ---
    if args.rms:
        return _loop_rms(penangkap, antrian_audio)

    # --- VAD ---
    try:
        from silero_vad import VADIterator, load_silero_vad

        vad = VADIterator(
            load_silero_vad(),
            sampling_rate=SAMPLE_RATE,
            threshold=args.vad_threshold,
            min_silence_duration_ms=400,
        )
    except Exception as exc:
        log.error("Gagal memuat Silero VAD: %s", exc)
        return 1

    # --- model emosi ---
    try:
        pengenal = PengenalEmosi(args.model)
    except Exception as exc:
        log.error("Gagal memuat model emosi: %s", exc)
        log.error("Butuh koneksi internet saat pertama kali (download ke cache HF).")
        return 1

    pekerja = threading.Thread(
        target=loop_fn,
        args=(pengenal, antrian_segmen, berhenti, args.threshold, args.smooth),
        daemon=True,
        name="inferensi",
    )
    pekerja.start()

    segmenter = SegmenterVAD(vad, window_sec=args.window, hop_sec=args.hop)
    sisa = np.empty(0, dtype=np.float32)

    print("[listening...]  (Ctrl+C untuk berhenti)", flush=True)

    try:
        with penangkap:
            while True:
                try:
                    blok = antrian_audio.get(timeout=0.5)
                except queue.Empty:
                    continue

                sisa = np.concatenate([sisa, resample(blok)])

                # Proses per frame 512 sampel sesuai kebutuhan Silero VAD.
                while len(sisa) >= VAD_FRAME:
                    frame, sisa = sisa[:VAD_FRAME], sisa[VAD_FRAME:]
                    ev = segmenter.proses(frame)

                    if ev.mulai:
                        print("\U0001f399  speech detected", flush=True)
                    if ev.audio is not None:
                        kirim_terbaru(antrian_segmen, ev.audio)
                    if ev.selesai:
                        kirim_terbaru(antrian_segmen, AKHIR)
    except KeyboardInterrupt:
        print("\n[berhenti]", flush=True)
    finally:
        berhenti.set()
        # Sentinel agar thread inferensi tidak menunggu sampai timeout.
        kirim_terbaru(antrian_segmen, None)
        pekerja.join(timeout=2.0)
        vad.reset_states()

    return 0


def _loop_rms(penangkap: PenangkapAudio, antrian_audio: queue.Queue) -> int:
    """Mode uji: cetak level RMS mic supaya jelas audio benar-benar masuk."""
    print(
        f"[rms] device={penangkap.device} samplerate={penangkap.samplerate} Hz"
        "  (Ctrl+C untuk berhenti)",
        flush=True,
    )
    try:
        with penangkap:
            while True:
                try:
                    blok = antrian_audio.get(timeout=0.5)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(np.square(blok))))
                bar = "#" * min(50, int(rms * 400))
                print(f"\rrms {rms:.4f} |{bar:<50}|", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[berhenti]", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deteksi emosi ucapan dari mic, output ke terminal."
    )
    p.add_argument("--device", type=int, default=None, help="Index device input mic.")
    p.add_argument("--list-devices", action="store_true", help="Tampilkan device audio.")
    p.add_argument("--rms", action="store_true", help="Mode uji: cetak level mic saja.")
    p.add_argument("--model", default=MODEL_ID, help="ID model SER di HuggingFace.")
    p.add_argument(
        "--threshold",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Ambang confidence untuk mencetak label (default {CONF_THRESHOLD}).",
    )
    p.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Sensitivitas Silero VAD, 0-1 (kecil = lebih sensitif).",
    )
    p.add_argument(
        "--window",
        type=float,
        default=WINDOW_SEC,
        help=f"Panjang jendela audio yang dinilai, detik (default {WINDOW_SEC}).",
    )
    p.add_argument(
        "--hop",
        type=float,
        default=HOP_SEC,
        help=f"Jarak antar-tebakan, detik. Kecil = lebih responsif (default {HOP_SEC}).",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=SMOOTH_ALPHA,
        help=f"Bobot tebakan baru, 0-1. 1.0 = tanpa smoothing (default {SMOOTH_ALPHA}).",
    )
    p.add_argument("--verbose", action="store_true", help="Log detail (debug).")
    return p.parse_args(argv)


def main(loop_fn: Optional[Callable] = None) -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.list_devices:
        try:
            import sounddevice as sd

            print(sd.query_devices())
        except Exception as exc:
            log.error("Gagal membaca device audio: %s", exc)
            return 1
        return 0

    return jalankan(args, loop_fn)


if __name__ == "__main__":
    raise SystemExit(main())
