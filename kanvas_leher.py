"""Kanvas yang bisa memiringkan/menganggukkan gambar kepala."""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any

from constants import CX, PIVOT_Y


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
