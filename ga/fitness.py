"""
LANGKAH 2 - FUNGSI FITNESS
  C = sum(HTM x orang) + sum(jarak km x Rp5.000 x kendaraan)    biaya total (Rp)
  D = sum(lama kunjungan) + sum(waktu tempuh)                   durasi total (jam)
  VALID  jika  C <= anggaran B  dan  D <= T = hari x jam per hari

  cost_score     = |C - B| / B
  duration_score = |D - T| / T
  penalti        = 10 x (kelebihan biaya relatif + kelebihan durasi relatif)
  fitness        = (1 + minat) / (0,01 + cost_score + duration_score + penalti)

Tanpa penalti dan bonus minat, rumusnya sama dengan paper acuan: 1 / (cost_score + duration_score).
"""
import math
from dataclasses import dataclass, field

from .data import DIKECUALIKAN

TAXI_RATE_PER_KM = 5000
SEATS_PER_VEHICLE = 4
PENALTY = 10
EPSILON = 0.01
# Aturan kode asli: sel -1 atau 0 (di luar diagonal) diganti 15 km / 1.300 detik.
FALLBACK_KM = 15
FALLBACK_HOURS = 1300 / 3600


@dataclass
class TripInput:
    budget: int = 1_000_000  # anggaran total rombongan (Rp)
    days: int = 5
    hours_per_day: float = 7
    people: int = 2
    tag_weights: dict = field(default_factory=dict)  # bobot minat per tag, mis. {"museum": 1, "history": 0.5}


def validate_input(trip):
    errors = []
    if not trip.budget > 0:
        errors.append("anggaran harus > 0")
    if int(trip.days) != trip.days or trip.days < 1:
        errors.append("jumlah hari minimal 1")
    if not 0 < trip.hours_per_day <= 24:
        errors.append("jam per hari harus 1 sampai 24")
    if int(trip.people) != trip.people or trip.people < 1:
        errors.append("jumlah orang minimal 1")
    if errors:
        raise ValueError("Masukan tidak valid: " + ", ".join(errors))


def preference_of(tags, weights):
    """Skor minat destinasi = bobot tertinggi di antara tag-nya / bobot tertinggi yang diisi pengguna."""
    top = max([0, *weights.values()])
    if top <= 0:
        return 0.0
    return max([0, *(weights.get(t, 0) for t in tags)]) / top


@dataclass
class Problem:
    """Masalah siap evaluasi. GA bekerja dengan indeks kandidat 0..n-1."""

    trip: TripInput
    destinasi: list  # semua destinasi di dataset
    candidates: list  # candidates[c] = indeks destinasi di dataset
    n: int
    ticket: list
    visit: list
    dist_km: list  # matriks n x n antar kandidat (km)
    travel_h: list  # matriks n x n antar kandidat (jam)
    preference: list  # skor minat tiap kandidat, 0 sampai 1
    target_hours: float  # T = hari x jam per hari
    vehicles: int


def build_problem(dataset, trip):
    validate_input(trip)
    cand = [d.index for d in dataset.destinasi if d.name not in DIKECUALIKAN]
    n = len(cand)
    dist_km = [[0.0] * n for _ in range(n)]
    travel_h = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(cand):
        for j, b in enumerate(cand):
            if i == j:
                continue
            m = dataset.distance_m[a][b]
            s = dataset.duration_s[a][b]
            dist_km[i][j] = FALLBACK_KM if m <= 0 else m / 1000
            travel_h[i][j] = FALLBACK_HOURS if s <= 0 else s / 3600
    dest = dataset.destinasi
    return Problem(
        trip=trip,
        destinasi=dest,
        candidates=cand,
        n=n,
        ticket=[dest[a].ticket for a in cand],
        visit=[dest[a].visit_hours for a in cand],
        dist_km=dist_km,
        travel_h=travel_h,
        preference=[preference_of(dest[a].tags, trip.tag_weights) for a in cand],
        target_hours=trip.days * trip.hours_per_day,
        vehicles=math.ceil(trip.people / SEATS_PER_VEHICLE),
    )


@dataclass
class Evaluation:
    total_cost: float
    total_hours: float
    distance_km: float
    cost_error: float  # (C - B) / B, negatif berarti di bawah anggaran
    duration_error: float  # (D - T) / T, negatif berarti di bawah batas jam
    preference: float  # rata-rata skor minat destinasi di rute
    valid: bool
    fitness: float


def evaluate_route(problem, route):
    p = problem
    ticket = sum(p.ticket[c] for c in route)
    visit = sum(p.visit[c] for c in route)
    pref = sum(p.preference[c] for c in route)
    distance = sum(p.dist_km[a][b] for a, b in zip(route, route[1:]))
    travel = sum(p.travel_h[a][b] for a, b in zip(route, route[1:]))

    total_cost = ticket * p.trip.people + distance * TAXI_RATE_PER_KM * p.vehicles
    total_hours = visit + travel
    B, T = p.trip.budget, p.target_hours
    cost_error = (total_cost - B) / B
    duration_error = (total_hours - T) / T
    preference = pref / len(route) if route else 0.0

    penalty = PENALTY * (max(0, cost_error) + max(0, duration_error))
    fitness = (1 + preference) / (EPSILON + abs(cost_error) + abs(duration_error) + penalty)
    return Evaluation(
        total_cost=total_cost,
        total_hours=total_hours,
        distance_km=distance,
        cost_error=cost_error,
        duration_error=duration_error,
        preference=preference,
        valid=total_cost <= B and total_hours <= T,
        fitness=fitness,
    )
