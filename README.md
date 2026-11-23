# SER Terminal

Dengarkan ucapan dari mic, lalu gerakkan **servo wajah** robot sesuai emosinya. Emosi ucapan dideteksi dengan *Speech Emotion Recognition*, diterjemahkan ke 5 ekspresi (senyum, netral, sedih, marah, kaget), lalu jadi pose untuk 11 servo wajah yang dikirim ke ESP32 lewat USB-serial. Tidak ada GUI; tanpa servo, pose dicetak ke terminal.

```
PC (mic → VAD → wav2vec2 → pose) --USB/UART--> ESP32 --I2C--> PCA9685 --> servo
```

ESP32 hanya **jembatan UART → I2C**: ia tidak mengolah data sedikit pun. Seluruh urusan PCA9685 (inisialisasi, 50 Hz, konversi nilai ke pulsa servo, kalibrasi) dikerjakan di PC; firmware ESP32 dibuat sendiri dan cukup meneruskan byte (lihat Protokol di bawah).

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
                     wav2vec2 SER → probabilitas label emosi
                              │
                     agregasi label → skor 5 ekspresi (+ EMA)
                              │
                     blend pose (7 gerakan → 11 servo) → ESP32 (UART)
```

Capture audio non-blocking: callback mic hanya menyalin blok ke antrian, sementara VAD dan inferensi dikerjakan di luar callback. Selama VAD bilang masih ada ucapan, tiap 0.3 detik model menilai 1.5 detik audio terakhir — jadi pose diperbarui selagi orang bicara, bukan menunggu kalimat selesai. Kalau inferensi tertinggal, jendela lama **dibuang** dan hanya yang terbaru dipakai.

## Struktur Kode

| File | Isi |
| --- | --- |
| `main.py` | Perakit: baca opsi CLI, pilih tujuan pose, jalankan pipeline |
| `pipeline.py` | `jalankan()`: mic → resample → VAD → thread inferensi; `uji_rms()` untuk tes mic |
| `audio.py` | `PenangkapAudio` (sounddevice, non-blocking) + resampler |
| `vad.py` | `SegmenterVAD`: Silero VAD → jendela ucapan |
| `model_emosi.py` | `PengenalEmosi` (wav2vec2 HuggingFace) + peta label → ekspresi |
| `pose.py` | Tabel pose (7 gerakan → 11 servo) + blend skor → pose |
| `pengendali.py` | `PengendaliEkspresi`: model → skor → pose → servo |
| `servo.py` | `PengirimServo`: UART ke ESP32 (tanpa `--port` = cetak pose ke terminal) |
| `konfigurasi.py` | Konstanta bersama (sample rate, ambang, jendela, dst.) |

Hanya lima class, masing-masing menyimpan state: `PenangkapAudio` (stream mic), `SegmenterVAD` (buffer ucapan), `PengenalEmosi` (model), `PengendaliEkspresi` (EMA dan pose terakhir), dan `PengirimServo` (port serial). Selebihnya fungsi biasa. Servo dikirim langsung dari thread inferensi, tanpa thread tambahan.

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
python main.py --list-ports                          # cari port ESP32
python main.py --port /dev/cu.usbserial-0001         # kirim pose ke servo (Windows: --port COM5)
python main.py                                       # tanpa servo: cetak pose ke terminal
```

Opsi:

| Opsi | Fungsi |
| --- | --- |
| `--port P` | Port serial ESP32. Kosong = hanya cetak pose |
| `--baud 115200` | Baud rate UART (samakan dengan `Serial.begin` di ESP32) |
| `--list-ports` | Tampilkan port serial lalu keluar |
| `--list-devices` | Tampilkan semua device audio beserta index-nya |
| `--device N` | Pilih mic tertentu berdasarkan index |
| `--rms` | Mode uji: cetak level mic saja, tanpa VAD/model/servo |
| `--threshold 0.4` | Ambang bawah gerbang keyakinan; di bawahnya pose memudar ke netral |
| `--vad-threshold 0.5` | Sensitivitas VAD (kecil = lebih sensitif) |
| `--hop 0.3` | Jarak antar-tebakan, detik. Kecil = lebih responsif, lebih berat |
| `--window 1.5` | Panjang audio yang dinilai tiap tebakan, detik |
| `--smooth 0.4` | Bobot tebakan baru, 0–1. `1.0` = tanpa smoothing (paling gesit tapi goyah) |
| `--model ID` | Ganti model SER dari HuggingFace |
| `--verbose` | Log detail (skor ekspresi dan pose tiap jendela) |

## Ekspresi dan Pose

Lima ekspresi. Label emosi model yang mirip **digabung** ke ekspresi yang sama — probabilitasnya dijumlahkan, jadi keraguan model antara mis. `fearful` dan `surprised` tidak menggoyang wajah:

| Ekspresi | Label emosi yang menyumbang |
| --- | --- |
| senyum | `happy` |
| netral | `calm`, `neutral` |
| sedih | `sad` |
| marah | `angry`, `disgust` |
| kaget | `fearful`, `surprised` |

Label yang tidak dikenal (model custom) otomatis dipetakan ke netral dengan peringatan di log, jadi ganti model tidak membuat program crash.

Skor **tidak** dipakai memilih satu pemenang, melainkan mem-blend pose. Tiap ekspresi punya pose penuh, dan pose akhir = jumlah pose berbobot skor, dikali gerbang keyakinan (`pose = gate × Σ skor × pose_penuh`). Ragu antara dua ekspresi = wajah campuran; makin yakin = amplitudo makin penuh; bingung total = netral. Setelah 2,5 detik tanpa ucapan, pose kembali netral.

Pose dihitung dalam **7 gerakan**, lalu dibagikan ke **11 servo** (kiri dan kanan mendapat nilai sama):

| Gerakan | Rentang | Arti | Servo (channel PCA9685) |
| --- | --- | --- | --- |
| buka mata | -10..10 | -10 merem, 0 normal, 10 melotot | 0 kelopak kiri, 1 kelopak kanan |
| angkat alis | -10..10 | -10 mengernyit, 10 naik | 2 alis kiri, 3 alis kanan |
| mulut x | -10..10 | -10 menguncup, 10 senyum lebar | 4 sudut mulut kiri, 5 sudut mulut kanan |
| mulut y | 0..10 | 0 mingkem, 10 menganga | 6 mulut buka |
| mata x | -10..10 | -10 melirik kiri, 10 kanan | 7 mata kiri x, 8 mata kanan x |
| mata y | -10..10 | -10 menunduk, 10 mendongak | 9 mata y |
| sudut mata | -10..10 | -10 miring sedih, 10 miring marah | 10 sudut mata |

Tabel pose per-ekspresi ada di `pose.py` (`POSE_PENUH`), disusun mengikuti kombinasi Action Unit **FACS** — itu titik awal yang wajar, silakan tuning ke karakter mekanikmu. Nilai `mata y` dan `sudut mata` per ekspresi (kaget mendongak, sedih menunduk, marah miring tajam) adalah **tebakan**, bukan hasil pengukuran; `mata x` sengaja 0 untuk semua ekspresi. Bobot koreksi per ekspresi (mis. `sedih` ×1.8 karena model cenderung menaruh bicara pelan di netral) ada di `pengendali.py` (`BOBOT_EKSPRESI`); itu juga heuristik.

## Protokol UART ke ESP32

ESP32 tidak mengolah data. PC mengirim perintah I2C yang sudah jadi, dibungkus bingkai biner (115200 baud 8N1):

```
[N] [alamat I2C] [byte data ...]        N = jumlah byte yang menyusul (alamat + data)
```

**Firmware ESP32, seluruhnya:**

1. baca 1 byte → `N`
2. baca `N` byte berikutnya ke buffer
3. `Wire.beginTransmission(buffer[0]); Wire.write(buffer + 1, N - 1); Wire.endTransmission();`
4. ulangi

Tidak perlu tahu apa itu PCA9685, servo, atau pose. Tidak ada balasan ke PC. Pastikan buffer cukup (bingkai terpanjang 47 byte; `Wire` ESP32 umumnya 128 byte; cek di versi core Anda) dan `Wire.begin()` serta `Serial.begin(115200)` dipanggil.

**Yang dikirim PC** (semuanya di [servo.py](servo.py)):

- Sekali tiap port tersambung: 4 bingkai inisialisasi PCA9685 (tidur, `PRESCALE` = 121 untuk 50 Hz, mode keluaran, bangun + auto-increment).
- Tiap kali pose berubah: satu bingkai 47 byte yang menulis 11 channel PCA9685 sekaligus (4 byte per channel: `ON_L, ON_H, OFF_L, OFF_H`). Tidak ada pengulangan kalau pose tetap.
- Saat program ditutup (Ctrl+C): bingkai pose netral.
- Servo ke-n dipasang di channel PCA9685 ke-(n-1), sesuai tabel di atas. Alamat I2C default `0x40`.

**Kalibrasi servo** ada di `KALIBRASI` di [servo.py](servo.py): untuk tiap servo, tick PWM pada nilai terendah dan tertinggi. Isi awalnya 205–410 tick (≈1,0–2,0 ms), aman untuk servo hobi umum tapi **belum dituning ke mekanikmu**. Sempitkan kalau servo menabrak batas mekanik, dan **tukar kedua angka** kalau servo terpasang berlawanan arah (mis. servo sisi kanan).

- Pose berubah langsung tanpa interpolasi, dikirim tiap ~0,3 dtk selama ada ucapan. Karena ESP32 tidak mengolah apa pun, gerak servo mengikuti kecepatan servonya sendiri; bila ingin lebih halus, perkecil `--hop` atau tambahkan penghalusan di PC.
- Membuka port me-reset ESP32, jadi program menunggu ±2 dtk sebelum mengirim. Kalau kabel tercabut, program menyambung ulang otomatis tiap 2 dtk dan menginisialisasi ulang PCA9685.
- Tanpa `--port`, yang dicetak adalah 7 gerakan, bukan bingkai UART.

## Model

Default: [`Dpngtm/wav2vec2-emotion-recognition`](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition), 7 label emosi (`angry`, `calm`, `disgust`, `fearful`, `happy`, `sad`, `surprised` — tanpa `neutral`; bicara datar jatuh ke `calm`, yang di sini dipetakan ke netral).

Alternatif 4 label (`neutral`, `happy`, `angry`, `sad`) yang sudah diuji jalan — ekspresi kaget tidak akan pernah muncul dengannya:

```bash
python main.py --model superb/wav2vec2-base-superb-er
```

> Hindari `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` meski config-nya menjanjikan 8 kelas. Pada `transformers` 5.x bobot classifier head-nya tidak cocok (`classifier.dense`/`classifier.output` versus `classifier.weight`), jadi head-nya di-inisialisasi acak dan prediksinya jadi sampah — tanpa error yang jelas.

## Yang Perlu Diketahui

**Akurasi SER pada bicara spontan rendah — apalagi bahasa Indonesia.** Model ini dilatih pada RAVDESS: rekaman aktor **berbahasa Inggris** yang *memerankan* emosi dengan ekspresi berlebihan dan kalimat seragam. Bicara Indonesia sehari-hari berbeda di dua lapis sekaligus (bahasa lain + spontan, bukan akting), jadi wajar kalau hasilnya terasa tidak nyambung dengan yang Anda rasakan. Pengelompokan ke 5 ekspresi meredam sebagian kekacauannya, tapi tetap perlakukan output sebagai **indikasi kasar**. Cara mengetes yang adil: bicara dengan emosi yang sengaja dilebih-lebihkan seperti sedang akting — itulah domain yang model ini kenal. Jangan dipakai untuk keputusan yang berdampak ke orang.

**Ada jeda sekitar 0,4–1,5 detik sebelum servo bergerak. Ini wajar, bukan bug.** Emosi tidak bisa dinilai dari audio yang terlalu pendek, jadi model baru menebak setelah terkumpul minimal 0.4 detik ucapan. Setelah itu pembaruan mengalir tiap 0.3 detik, dan tebakan awal lebih goyah karena jendela pertama baru berisi ~0.6 detik audio — itu alasan smoothing ada.

**Peredam bekerja berlapis.** (1) penjumlahan probabilitas label serumpun, (2) EMA `--smooth` (pada rekaman uji 15 detik: tanpa smoothing label berganti 7×/27 jendela, di `0.4` jadi 5×, di `0.2` jadi 3×), (3) gerbang keyakinan yang menekan pose ke netral saat model bingung. Jalankan dengan `--verbose` untuk melihat skor di balik tiap pose.

**Model butuh input 16 kHz.** Salah sample rate = hasil ngawur, bukan sekadar kurang akurat. Program mencoba membuka mic langsung di 16 kHz; kalau device tidak mendukung, ia memakai sample rate bawaan device lalu me-resample ke 16 kHz otomatis (`torchaudio`). Jalankan dengan `--verbose` untuk melihat rate mana yang dipakai, dan `--list-devices` untuk memilih mic.

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

VAD terlalu ketat untuk mic Anda. Turunkan ambangnya dengan `--vad-threshold 0.3`. Sebaliknya, kalau "speech detected" muncul terus padahal sunyi (mic berisik), naikkan ke `0.6`–`0.7`.

**Servo tidak bergerak / `Gagal membuka port servo`**

Cek `python main.py --list-ports` dan pastikan port benar, kabel USB data (bukan kabel charge-only), dan tidak ada aplikasi lain (Serial Monitor Arduino IDE) yang memegang port. Tanpa `--port`, pose hanya dicetak; kalau baris pose tercetak tapi servo diam, masalahnya di firmware ESP32/PCA9685.

**Terasa kurang responsif**

```bash
python main.py --hop 0.2 --window 1.0 --smooth 0.7
```

Konsekuensinya: tebakan lebih sering tapi lebih goyah, dan CPU lebih sibuk. Kalau inferensi tidak sanggup mengejar hop, program otomatis membuang jendela basi — output tetap terkini, hanya beberapa pembaruan terlewat.

**Pose terlalu gampang berganti**

```bash
python main.py --smooth 0.2 --window 2.0
```

Sebaliknya, kalau pose terlalu lengket, naikkan `--smooth` (mis. `0.7`).

**Gagal membuka mic / `PortAudioError`**

Device sedang dipakai aplikasi lain (Zoom, Meet, perekam), atau index device salah. Tutup aplikasi tersebut lalu ulangi, dan verifikasi index dengan `--list-devices`.

**Inferensi lambat**

Default CPU-only. GPU dipakai otomatis kalau `torch.cuda.is_available()` bernilai true — tidak wajib, dan tidak perlu konfigurasi apa pun.
