"""Konstanta bersama untuk pipeline audio -> VAD -> model emosi."""

SAMPLE_RATE = 16_000
"""Sample rate wajib untuk Silero VAD maupun model wav2vec2."""

VAD_FRAME = 512
"""Silero VAD pada 16 kHz hanya menerima frame tepat 512 sampel (32 ms)."""

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"
"""Model SER siap-pakai (dilatih di RAVDESS, 7 kelas — tanpa 'neutral')."""

CONF_THRESHOLD = 0.40
"""Ambang confidence minimal; di bawah ini pose memudar ke netral."""

MIN_SPEECH_SEC = 0.4
"""Audio minimal sebelum model boleh menebak (di bawah ini terlalu pendek)."""

WINDOW_SEC = 1.5
"""Panjang jendela geser: emosi dinilai dari 1.5 detik ucapan terakhir."""

HOP_SEC = 0.3
"""Jarak antar-inferensi. Tebakan diperbarui tiap 0.3 detik selama bicara."""

PREROLL_SEC = 0.3
"""Audio sebelum VAD memicu 'start', supaya awal kata tidak terpotong."""

SMOOTH_ALPHA = 0.4
"""Bobot tebakan baru saat smoothing (1.0 = tanpa smoothing)."""
