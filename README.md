# SER Terminal

Dengarkan ucapan dari mic, log **ekspresi wajah** untuk robot mekanis — langsung di terminal. Emosi ucapan dideteksi dengan *Speech Emotion Recognition*, lalu diterjemahkan ke 5 ekspresi wajah yang masing-masing nantinya jadi satu preset pose servo. Program ini hanya me-log ekspresi; terjemahan ke gerakan aktuator dikerjakan terpisah.

Ekspresi diperbarui **selama orang masih bicara**, bukan menunggu kalimat selesai. Baris terakhir ditimpa di tempat tiap 0.3 detik; tiap kali ekspresi berganti (atau frasa berhenti), baris itu dikunci sebagai riwayat:

```
[listening...]  (Ctrl+C untuk berhenti)
🎙  speech detected
😐  netral   (conf 0.95)     ← baris ini terus berubah sementara orang bicara
🎙  speech detected
🙁  sedih    (conf 0.90)
```

## Cara Kerja

```
mic → callback sounddevice → antrian audio
                              │
                     resample ke 16 kHz
                              │
                     Silero VAD (deteksi ada/tidaknya ucapan)
                              │
                     jendela geser: 1.5 dtk terakhir, tiap 0.3 dtk
                              │
                     antrian → thread inferensi
                              │
                     wav2vec2 SER → probabilitas 7 label emosi
                              │
                     agregasi label → skor 5 ekspresi wajah
                              │
                     EMA + histeresis → print ekspresi ke terminal
```

Capture audio bersifat non-blocking: callback mic hanya menyalin blok ke antrian, sementara VAD dan inferensi model dikerjakan di luar callback.

**Jendela geser, bukan tunggu-selesai.** Selama VAD bilang masih ada ucapan, tiap 0.3 detik model menilai 1.5 detik audio terakhir. Jadi satu frasa 4 detik menghasilkan belasan tebakan berturut-turut, bukan satu tebakan di akhir. Buffer-nya dipangkas sepanjang jendela, jadi pemakaian memori tetap datar berapa lama pun Anda bicara.

Kalau inferensi sempat tertinggal dari suara, jendela lama **dibuang** dan hanya yang terbaru dipakai — output lebih baik melompat daripada tertinggal makin jauh di belakang.

## Install

Butuh Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Di Linux, `sounddevice` butuh PortAudio:

```bash
sudo apt install libportaudio2
```

Jalan pertama kali butuh koneksi internet untuk mengunduh model (~360 MB, disimpan di cache HuggingFace `~/.cache/huggingface`). Setelah itu bisa offline — kalau HuggingFace tidak bisa dihubungi, program otomatis jatuh ke cache lokal.

## Jalankan

```bash
python main.py
```

Opsi:

| Opsi | Fungsi |
| --- | --- |
| `--list-devices` | Tampilkan semua device audio beserta index-nya |
| `--device N` | Pilih mic tertentu berdasarkan index |
| `--rms` | Mode uji: cetak level mic saja, tanpa VAD/model |
| `--threshold 0.4` | Skor minimal agar ekspresi boleh dipasang/diganti; di bawahnya wajah bertahan |
| `--vad-threshold 0.5` | Sensitivitas VAD (kecil = lebih sensitif) |
| `--hop 0.3` | Jarak antar-tebakan, detik. Kecil = lebih responsif, lebih berat |
| `--window 1.5` | Panjang audio yang dinilai tiap tebakan, detik |
| `--smooth 0.4` | Bobot tebakan baru, 0–1. `1.0` = tanpa smoothing (paling gesit tapi berkedip) |
| `--model ID` | Ganti model SER dari HuggingFace |
| `--verbose` | Log detail |

## Ekspresi Wajah

Lima ekspresi, masing-masing calon satu preset pose servo. Label emosi model yang mirip **digabung** ke ekspresi yang sama — probabilitasnya dijumlahkan, jadi keraguan model antara mis. `fearful` dan `surprised` tidak menggoyang wajah:

| Ekspresi | Label emosi yang menyumbang |
| --- | --- |
| 🙂 senyum | `happy` |
| 😐 netral | `calm`, `neutral` |
| 🙁 sedih | `sad` |
| 😠 marah | `angry`, `disgust` |
| 😲 kaget | `fearful`, `surprised` |

Dua aturan penting untuk wajah robot:

- **Histeresis.** Ekspresi berganti hanya kalau pemenang baru menang telak (skor ≥ 0.6) atau menang dua jendela berturut-turut (~0.6 dtk). Terukur pada rekaman uji 15 detik: label emosi mentah berganti 5×, ekspresi hanya 2× — dan keduanya pergantian yang sungguhan, bukan kedip.
- **Wajah bertahan saat hening.** Saat orang berhenti bicara, ekspresi terakhir tetap "terpasang" — tidak di-reset ke netral. Kapan robot kembali ke pose istirahat adalah keputusan di sisi penggerak servo, bukan di sini.

Label yang tidak dikenal (model custom) otomatis dipetakan ke netral dengan peringatan di log, jadi ganti model tidak pernah membuat program crash.

## main2.py — Output Derajat Pose (untuk servo)

Pipeline sama, tapi langkah terakhirnya berbeda: alih-alih memilih satu ekspresi pemenang, `main2.py` **mem-blend pose**. Tiap ekspresi punya pose penuh dalam 4 derajat kebebasan, dan pose akhir = jumlah pose berbobot skor model, dikali gerbang keyakinan (`pose = gate × Σ skor × pose_penuh`). Ragu antara dua ekspresi = wajah campuran; makin yakin = amplitudo makin penuh; bingung total = netral.

```bash
python main2.py
```

Satu baris per jendela (~tiap 0.3 dtk selama ada ucapan) ke stdout:

```
buka mata -2; angkat alis 0; mulut x 5; mulut y 2
```

| DOF | Rentang | Arti |
| --- | --- | --- |
| buka mata | -10..10 | -10 full merem, 0 normal, 10 full melotot |
| angkat alis | -10..10 | -10 mengernyit, 0 normal, 10 naik full |
| mulut x | -10..10 | -10 menguncup, 0 normal, 10 senyum lebar |
| mulut y | 0..10 | 0 mingkem, 10 menganga lebar |

Parser di sisi servo cukup ambil baris stdout yang diawali `buka mata` (baris lain hanya banner/status). Tabel pose per-ekspresi ada di atas `main2.py` (`POSE_PENUH`), disusun mengikuti kombinasi Action Unit **FACS** — itu titik awal yang wajar, silakan tuning ke karakter mekanikmu. Karena outputnya kontinu, tidak ada histeresis; pergerakan antar-baris sudah halus (terukur: lonjakan terbesar 3.6 satuan per langkah). Saat hening, baris berhenti mengalir dan pose terakhir dianggap tetap terpasang.

Semua opsi `main.py` berlaku (`--threshold` menjadi ambang bawah gerbang keyakinan).

## Model

Default: [`Dpngtm/wav2vec2-emotion-recognition`](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition), 7 label emosi (`angry`, `calm`, `disgust`, `fearful`, `happy`, `sad`, `surprised` — tanpa `neutral`; bicara datar jatuh ke `calm`, yang di sini memang dipetakan ke ekspresi 😐 netral).

Alternatif 4 label (`neutral`, `happy`, `angry`, `sad`) yang sudah diuji jalan — ekspresi 😲 kaget tidak akan pernah muncul dengannya:

```bash
python main.py --model superb/wav2vec2-base-superb-er
```

> Hindari `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` meski config-nya menjanjikan 8 kelas. Pada `transformers` 5.x bobot classifier head-nya tidak cocok (`classifier.dense`/`classifier.output` versus `classifier.weight`), jadi head-nya di-inisialisasi acak dan prediksinya jadi sampah — tanpa error yang jelas.

## Yang Perlu Diketahui

**Akurasi SER pada bicara spontan rendah — apalagi bahasa Indonesia.** Model ini dilatih pada RAVDESS: rekaman aktor **berbahasa Inggris** yang *memerankan* emosi dengan ekspresi berlebihan dan kalimat seragam. Bicara Indonesia sehari-hari berbeda di dua lapis sekaligus (bahasa lain + spontan, bukan akting), jadi wajar kalau hasil tes terasa tidak nyambung dengan yang Anda rasakan. Pengelompokan ke 5 ekspresi meredam sebagian besar kekacauannya, tapi tetap perlakukan output sebagai **indikasi kasar**. Cara mengetes yang adil: bicara dengan emosi yang sengaja dilebih-lebihkan seperti sedang akting — itulah domain yang model ini kenal. Jangan dipakai untuk keputusan yang berdampak ke orang.

**Tebakan pertama tetap butuh ~0.4 detik. Ini wajar, bukan bug.** Emosi tidak bisa dinilai dari audio yang terlalu pendek, jadi model baru menebak setelah terkumpul minimal 0.4 detik ucapan (diukur: tebakan pertama muncul 0.35 dtk setelah bicara dimulai). Setelah itu barulah pembaruan mengalir tiap 0.3 detik. Inferensi sendiri ~110 ms per jendela di CPU, jadi masih longgar terhadap hop 300 ms.

**Tebakan awal lebih goyah.** Jendela pertama cuma berisi ~0.6 detik audio, dan model memang tidak stabil di potongan sependek itu — label sering meleset di awal lalu mengendap setelah jendela penuh 1.5 detik. Ini alasan smoothing ada.

**Tiga lapis peredam bekerja bersama.** Menilai potongan 1.5 detik satu per satu memang bising — label emosi mentah bisa berganti-ganti tiap jendela. Yang menjaga wajah tetap tenang: (1) penjumlahan probabilitas label serumpun, (2) EMA `--smooth` (pada rekaman uji: tanpa smoothing label berganti 7×/27 jendela, di `0.4` jadi 5×, di `0.2` jadi 3×), lalu (3) histeresis pergantian ekspresi. Jalankan dengan `--verbose` kalau ingin melihat skor mentah per ekspresi di balik keputusan.

**Model butuh input 16 kHz.** Salah sample rate = hasil ngawur, bukan sekadar kurang akurat. Program mencoba membuka mic langsung di 16 kHz; kalau device tidak mendukung, ia memakai sample rate bawaan device lalu me-resample ke 16 kHz secara otomatis (`torchaudio`). Jalankan dengan `--verbose` untuk melihat rate mana yang dipakai.

## Troubleshooting

**Tidak ada output sama sekali / mic tidak kedengaran**

Cek dulu audio benar-benar masuk:

```bash
python main.py --rms
```

Bicaralah — angka RMS harus naik (biasanya di atas ~0.01) dan bar bergerak. Kalau tetap 0.0000, mic-nya yang salah, bukan modelnya.

**Salah mic terpilih**

```bash
python main.py --list-devices
python main.py --device 2
```

Index device di sisi input adalah angka di kolom paling kiri. Di macOS, pastikan aplikasi terminal Anda punya izin mikrofon (System Settings → Privacy & Security → Microphone).

**"speech detected" tidak pernah muncul padahal RMS naik**

VAD terlalu ketat untuk mic Anda. Turunkan ambangnya:

```bash
python main.py --vad-threshold 0.3
```

Sebaliknya, kalau "speech detected" muncul terus padahal sunyi (mic berisik), naikkan ke `0.6`–`0.7`.

**Muncul `🤷 (belum yakin: ...)` di awal**

Belum ada ekspresi terpasang dan skor tertinggi masih di bawah `--threshold`. Hanya terjadi di ucapan pertama; setelah ada ekspresi terpasang, skor rendah membuat wajah bertahan, bukan jadi tanda tanya. Turunkan `--threshold` kalau ingin ekspresi pertama terpasang lebih cepat.

**Terasa masih kurang responsif**

Perkecil hop dan jendelanya:

```bash
python main.py --hop 0.2 --window 1.0 --smooth 0.7
```

Konsekuensinya: tebakan lebih sering tapi lebih goyah, dan CPU lebih sibuk. Kalau inferensi tidak sanggup mengejar hop, program otomatis membuang jendela basi — output tetap terkini, hanya saja beberapa pembaruan terlewat.

**Ekspresi masih terasa gampang berganti**

Perbesar peredamannya (angka lebih kecil = lebih kalem), atau perpanjang jendelanya:

```bash
python main.py --smooth 0.2 --window 2.0
```

Sebaliknya, kalau wajah terasa terlalu lengket pada satu ekspresi, naikkan `--smooth` (mis. `0.7`).

**Gagal membuka mic / `PortAudioError`**

Device sedang dipakai aplikasi lain (Zoom, Meet, perekam), atau index device salah. Tutup aplikasi tersebut lalu ulangi, dan verifikasi index dengan `--list-devices`.

**Inferensi lambat**

Default CPU-only. GPU dipakai otomatis kalau `torch.cuda.is_available()` bernilai true — tidak wajib, dan tidak perlu konfigurasi apa pun.
