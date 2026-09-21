# CLAUDE.md — SER Terminal Sederhana

## Tujuan

Aplikasi Python yang berjalan di **terminal**. PC mendengarkan dari **mic / audio input** secara terus-menerus, dan setiap kali ada ucapan, cetak **emosi yang sedang berlangsung** ke terminal. Tidak ada GUI, tidak ada grafik, tidak ada tombol. Cukup jalan → dengar → cetak label emosi.

Contoh output yang diinginkan:

```
[listening...]
🎙  speech detected
😊  senang        (conf 0.72)
🎙  speech detected
😐  netral        (conf 0.55)
🎙  speech detected
😠  marah         (conf 0.68)
```

## Prinsip: SEDERHANA

Utamakan kesederhanaan di atas segalanya. Jangan over-engineer. Satu file `main.py` boleh saja jika lebih jelas. Tidak perlu abstraksi berlebihan, tidak perlu web server, tidak perlu output dimensional A/V/D — cukup **label emosi kategorikal** (mis. senang, sedih, marah, netral, takut, dst. sesuai model).

## Batasan Teknis

- **Python 3.10+**, komentar/docstring Bahasa Indonesia.
- **Capture audio non-blocking**: `sounddevice` dengan callback → buffer. Inferensi di thread terpisah lewat `queue.Queue`. Capture jangan ter-block oleh model.
- **Sample rate 16 kHz mono**. Resample otomatis jika device beda. Ini wajib — salah sample rate = hasil ngawur.
- **VAD**: pakai **Silero VAD** agar model hanya jalan saat ada ucapan (jangan proses keheningan). Ambil segmen ucapan berdasarkan batas dari VAD.
- **Model emosi**: gunakan model wav2vec2 SER siap-pakai dari HuggingFace `transformers` (mis. `Dpngtm/wav2vec2-emotion-recognition` atau setara) yang langsung output label + confidence. Load sekali di awal.
- **CPU-only sebagai default.** Boleh pakai GPU jika tersedia (deteksi otomatis `torch.cuda.is_available()`), tapi jangan wajibkan.
- **Smoothing ringan**: agar output tidak berkedip, tahan/print label hanya jika confidence di atas ambang (mis. 0.4) atau jika label bertahan >1 segmen. Sederhana saja.

## Alur

```
mic → sounddevice callback → buffer
                              │
                     Silero VAD (deteksi ucapan)
                              │  (segmen saat user selesai satu frasa)
                              ▼
                     queue → thread inferensi
                              │
                     wav2vec2 SER → (label, confidence)
                              │
                          print ke terminal
```

## Struktur

Cukup minimal:

```
ser_terminal/
├── README.md
├── requirements.txt
└── main.py          # semua di sini: capture + VAD + model + loop print
```

Boleh dipecah jadi 2-3 file jika benar-benar lebih jelas, tapi jangan lebih dari itu.

## Cara Kerja yang Diharapkan (bertahap, uji tiap langkah)

1. `requirements.txt` + `main.py` kerangka. Uji: mic tertangkap, cetak RMS level agar tahu audio masuk.
2. Tambah Silero VAD. Uji: cetak "speech detected" saat bicara, diam saat sunyi.
3. Load model wav2vec2, sambungkan: setiap segmen ucapan → prediksi → print label + confidence.
4. Tambah ambang confidence + graceful shutdown (Ctrl+C menutup stream dengan bersih).
5. Tulis README singkat (install, jalankan, troubleshooting mic & sample rate).

## Standar Kode

- Type hints, logging sederhana (bukan print liar untuk hal internal; print hanya untuk output emosi ke user).
- Thread-safe via queue. Ctrl+C bersih.
- Tangani error umum: mic tidak ada, model gagal load, sample rate mismatch.

## Catatan yang HARUS ada di README

- Akurasi SER pada bicara spontan **rendah** dibanding benchmark dataset akting — perlakukan output sebagai indikasi kasar, bukan pembacaan presisi.
- Ada jeda ~0.5–1.5 dtk karena emosi butuh durasi bicara; wajar, bukan bug.
- Model butuh input 16 kHz; jelaskan cara cek/pilih device mic.

## JANGAN

- Jangan bikin GUI, grafik, atau web server.
- Jangan capture audio yang blocking.
- Jangan output dimensional A/V/D — cukup label.
- Jangan tambah dependensi yang tidak dipakai.
- Jangan klaim akurasi tinggi.

---

**Mulai dari langkah 1.** Lanjutkan tanpa banyak bertanya kecuali ada yang benar-benar ambigu.
