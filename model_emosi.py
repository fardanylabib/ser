"""Model emosi wav2vec2 (HuggingFace) + pemetaan label model -> 5 ekspresi wajah."""

import logging
from typing import Any

import numpy as np

from konfigurasi import MODEL_ID, SAMPLE_RATE

log = logging.getLogger("ser.model")

# Peta label emosi model -> ekspresi wajah; probabilitas label serumpun dijumlahkan.
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

# Urutan label RAVDESS, fallback kalau config hanya berisi LABEL_n.
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


def peta_ekspresi(labels: list[str]) -> list[str]:
    """Bangun peta index-label -> nama ekspresi; label asing jadi 'netral'."""
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

    # Checkpoint tanpa nama label asli.
    if labels and all(nama.upper().startswith("LABEL_") for nama in labels):
        if len(labels) == len(RAVDESS_ORDER):
            log.warning("Config model tanpa nama label, memakai urutan RAVDESS.")
            return list(RAVDESS_ORDER)
    return labels


class PengenalEmosi:
    """Bungkus model wav2vec2 SER dari HuggingFace, di-load sekali di awal."""

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
            # Jaringan bermasalah, coba cache HF.
            log.warning("Gagal menghubungi HuggingFace (%s), coba pakai cache...", exc)
            self.extractor = AutoFeatureExtractor.from_pretrained(
                model_id, local_files_only=True
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id, local_files_only=True
            )

        self.model.to(self.device).eval()
        self.labels = rapikan_nama_label(self.model.config)
        self.peta = peta_ekspresi(self.labels)
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
