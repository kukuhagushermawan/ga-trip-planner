"""
LANGKAH 0 - DATA
Data asli dari repositori GA-pairwise-recommender-system (Qomariyah & Kazakov, 2021):
  jsonAttraction.txt  201 destinasi Jabodetabek (HTM Rp/orang, lama kunjungan jam, tag, koordinat)
  distanceMatrix.csv  jarak antardestinasi dalam meter  (-1 = tidak ada rute)
  durationMatrix.csv  waktu tempuh antardestinasi dalam detik  (-1 = tidak ada rute)
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Hasil audit data: 8 destinasi ini tidak dipakai sebagai kandidat GA.
SALAH_GEOKODE = "jarak > 100 km ke hampir semua destinasi (salah geokode)"
DIKECUALIKAN = {
    "East Java Pavilion": SALAH_GEOKODE,
    "Merdeka Palace": SALAH_GEOKODE,
    "Museum of East Timor": SALAH_GEOKODE,
    "National History Museum": SALAH_GEOKODE,
    "Pasar Seni": SALAH_GEOKODE,
    "Sangga Buana": SALAH_GEOKODE,
    "Sarmiento Parakantja Timber": SALAH_GEOKODE,
    "Nofri kasur": "tidak punya rute ke destinasi lain (semua sel -1)",
}


@dataclass(frozen=True)
class Destinasi:
    id: str  # ID001, ID002, ... sesuai urutan berkas
    index: int
    name: str
    lat: float
    lng: float
    tags: tuple
    ticket: int  # harga tiket masuk, Rp per orang
    visit_hours: float  # lama kunjungan, jam


@dataclass
class Dataset:
    destinasi: list
    distance_m: list  # matriks n x n, baris = asal, kolom = tujuan
    duration_s: list


def _angka(nilai, nama, baris):
    try:
        x = float(nilai)
    except (TypeError, ValueError):
        raise ValueError(f"destinasi ke-{baris}: {nama} '{nilai}' bukan angka") from None
    if x < 0:
        raise ValueError(f"destinasi ke-{baris}: {nama} tidak boleh negatif")
    return x


def parse_destinasi(text):
    places = json.loads(text).get("places")
    if not isinstance(places, list):
        raise ValueError('berkas destinasi tidak punya field "places"')
    hasil = []
    for i, p in enumerate(places):
        name = p.get("name", "")
        tags = p.get("tags") or []
        if not name or not tags:
            raise ValueError(f"destinasi ke-{i + 1}: nama atau tag kosong")
        loc = p["geometry"]["location"]
        hasil.append(
            Destinasi(
                id=f"ID{i + 1:03d}",
                index=i,
                name=name,
                lat=float(loc["lat"]),
                lng=float(loc["lng"]),
                tags=tuple(tags),
                ticket=int(_angka(p.get("ticketPrice"), "harga tiket", i + 1)),
                visit_hours=_angka(p.get("visitDuration"), "lama kunjungan", i + 1),
            )
        )
    return hasil


def parse_matriks(text):
    """Matriks: header `origin,<nama...>`, lalu satu baris per asal."""
    rows = [r for r in csv.reader(text.splitlines()) if r]
    labels = rows[0][1:]
    n = len(labels)
    if len(rows) - 1 != n:
        raise ValueError(f"matriks tidak persegi: {len(rows) - 1} baris, {n} kolom")
    cells = [[int(v) for v in row[1 : n + 1]] for row in rows[1:]]
    return labels, cells


def build_dataset(attractions_text, distance_text, duration_text):
    """
    Menggabungkan destinasi dan kedua matriks. Matriks dicocokkan berdasarkan NAMA (koma dibuang),
    karena urutan barisnya berbeda dengan urutan destinasi (ada label tambahan "Jakarta").
    """
    destinasi = parse_destinasi(attractions_text)

    def urutkan(text):
        labels, cells = parse_matriks(text)
        pos = {name: i for i, name in enumerate(labels)}
        try:
            idx = [pos[d.name.replace(",", "")] for d in destinasi]
        except KeyError as e:
            raise ValueError(f"{e} tidak ada di matriks") from None
        return [[cells[r][c] for c in idx] for r in idx]

    return Dataset(destinasi, urutkan(distance_text), urutkan(duration_text))


def load_dataset(folder=DATA_DIR):
    folder = Path(folder)
    read = lambda name: (folder / name).read_text(encoding="utf-8")
    return build_dataset(read("jsonAttraction.txt"), read("distanceMatrix.csv"), read("durationMatrix.csv"))
