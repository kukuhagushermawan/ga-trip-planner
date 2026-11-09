"""
LANGKAH 1 - REPRESENTASI KROMOSOM
Masalahnya gabungan Knapsack (memilih destinasi) dan TSP (mengurutkan rute), jadi kromosom punya dua bagian:

  mask  = [0,1,0,1,1]   kandidat 1, 3, 4 dipilih        (bagian pemilihan)
  order = [4,0,3,1,2]   urutan relatif semua kandidat   (bagian urutan)
  rute  = isi `order` yang mask-nya 1  ->  4, 3, 1

Setiap kromosom pasti terdekode menjadi rute tanpa destinasi ganda, jadi tidak perlu proses perbaikan.
"""
from dataclasses import dataclass


@dataclass
class Chromosome:
    mask: list  # 0/1 per kandidat
    order: list  # permutasi 0..n-1


def decode(c):
    """Kromosom -> rute (daftar indeks kandidat berurutan)."""
    return [g for g in c.order if c.mask[g] == 1]


def write_route(c, route):
    """Menulis urutan baru untuk gen yang sedang dipilih; posisi gen lain di `order` tidak berubah."""
    k = 0
    for i, g in enumerate(c.order):
        if c.mask[g] == 1:
            c.order[i] = route[k]
            k += 1


def clone(c):
    return Chromosome(c.mask[:], c.order[:])


def random_chromosome(n, k, rng):
    """Kromosom acak: `k` kandidat dipilih, urutan acak."""
    mask = [0] * n
    for g in rng.sample(range(n), min(k, n)):
        mask[g] = 1
    order = list(range(n))
    rng.shuffle(order)
    return Chromosome(mask, order)
