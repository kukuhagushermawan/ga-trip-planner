"""LANGKAH 3 - OPERATOR GENETIKA: seleksi, crossover, mutasi."""
from .kromosom import decode, write_route

# ---- Seleksi ----------------------------------------------------------------------------------


def tournament_select(fitness, k, rng):
    """Turnamen: ambil k individu acak, yang fitness-nya tertinggi menjadi induk."""
    best = rng.randrange(len(fitness))
    for _ in range(k - 1):
        challenger = rng.randrange(len(fitness))
        if fitness[challenger] > fitness[best]:
            best = challenger
    return best


# ---- Crossover --------------------------------------------------------------------------------


def uniform_crossover(a, b, rng):
    """Uniform crossover untuk `mask`: setiap bit ditukar dengan peluang 1/2."""
    c1, c2 = a[:], b[:]
    for i in range(len(a)):
        if rng.random() < 0.5:
            c1[i], c2[i] = b[i], a[i]
    return c1, c2


def order_crossover(p1, p2, i, j):
    """
    Order crossover (OX) untuk `order`: segmen [i..j] disalin dari induk 1, sisanya diisi gen induk 2
    sesuai urutan kemunculannya mulai setelah j. Anak selalu permutasi yang sah.
      P1 = 1 2 3 | 4 5 6 7 | 8 9,  P2 = 4 5 2 | 1 8 7 6 | 9 3   ->   anak = 2 1 8 | 4 5 6 7 | 9 3
    """
    n = len(p1)
    child = [None] * n
    child[i : j + 1] = p1[i : j + 1]
    taken = set(p1[i : j + 1])
    sisa = [p2[(j + 1 + s) % n] for s in range(n)]
    sisa = [g for g in sisa if g not in taken]
    for s, g in enumerate(sisa):
        child[(j + 1 + s) % n] = g
    return child


def order_crossover_pair(p1, p2, rng):
    i, j = sorted((rng.randrange(len(p1)), rng.randrange(len(p1))))
    return order_crossover(p1, p2, i, j), order_crossover(p2, p1, i, j)


# ---- Mutasi -----------------------------------------------------------------------------------


def swap_mutation(c, rng):
    """Swap: tukar posisi dua destinasi di rute."""
    route = decode(c)
    a, b = rng.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    write_route(c, route)


def insert_mutation(c, rng):
    """Insert: pindahkan satu destinasi ke posisi lain."""
    route = decode(c)
    a, b = rng.sample(range(len(route)), 2)
    g = route.pop(a)
    route.insert(b, g)
    write_route(c, route)


def add_mutation(c, rng):
    """Tambah: pilih satu kandidat yang belum ada di rute."""
    unselected = [g for g, bit in enumerate(c.mask) if bit == 0]
    c.mask[rng.choice(unselected)] = 1


def remove_mutation(c, rng):
    """Hapus: buang satu destinasi dari rute."""
    c.mask[rng.choice(decode(c))] = 0


def mutate(c, rate, rng):
    """
    Untuk setiap destinasi di rute (minimal sekali), dengan peluang `rate` dijalankan satu mutasi
    yang dipilih acak di antara yang bisa diterapkan.
    """
    applied = []
    for _ in range(max(1, sum(c.mask))):
        if rng.random() >= rate:
            continue
        length = sum(c.mask)
        options = []
        if length >= 2:
            options += ["swap", "insert"]
        if length < len(c.mask):
            options.append("tambah")
        if length >= 1:
            options.append("hapus")
        kind = rng.choice(options)
        {"swap": swap_mutation, "insert": insert_mutation, "tambah": add_mutation, "hapus": remove_mutation}[kind](c, rng)
        applied.append(kind)
    return applied
