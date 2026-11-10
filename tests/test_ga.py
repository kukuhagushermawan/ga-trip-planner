import random

import pytest

from ga.algoritma import GAConfig, run_ga
from ga.data import DIKECUALIKAN, load_dataset
from ga.fitness import TripInput, build_problem, evaluate_route, preference_of
from ga.kromosom import Chromosome, decode, random_chromosome, write_route
from ga.operators import mutate, order_crossover, uniform_crossover

dataset = load_dataset()
problem = build_problem(dataset, TripInput())


# ---- data ----


def test_memuat_201_destinasi_dan_matriks_201x201():
    assert len(dataset.destinasi) == 201
    assert len(dataset.distance_m) == 201 and len(dataset.distance_m[0]) == 201


def test_mengeluarkan_8_destinasi_sehingga_tersisa_193_kandidat():
    assert len(DIKECUALIKAN) == 8
    assert problem.n == 193


# ---- kromosom ----


def test_decode_contoh_di_slide():
    c = Chromosome([0, 1, 0, 1, 1], [4, 0, 3, 1, 2])
    assert decode(c) == [4, 3, 1]


def test_write_route_hanya_mengubah_gen_yang_dipilih():
    c = Chromosome([0, 1, 0, 1, 1], [4, 0, 3, 1, 2])
    write_route(c, [1, 4, 3])
    assert decode(c) == [1, 4, 3]
    assert c.order == [1, 0, 4, 3, 2]


# ---- fitness ----


def test_rute_kosong_fitness_rendah():
    assert evaluate_route(problem, []).fitness < 1


def test_biaya_dan_durasi_dihitung_benar():
    e = evaluate_route(problem, [0, 1])
    ticket = (problem.ticket[0] + problem.ticket[1]) * 2
    taxi = problem.dist_km[0][1] * 5000
    assert e.total_cost == pytest.approx(ticket + taxi)
    assert e.total_hours == pytest.approx(problem.visit[0] + problem.visit[1] + problem.travel_h[0][1])


def test_rencana_melanggar_batas_kena_penalti():
    e = evaluate_route(problem, list(range(60)))
    assert not e.valid
    assert e.fitness < 0.1


def test_skor_minat_dinormalisasi():
    assert preference_of(["museum", "outdoors"], {"museum": 0.5, "history": 1}) == 0.5
    assert preference_of(["beach"], {}) == 0


# ---- operator ----


def test_order_crossover_contoh_davis_1985():
    p1 = [x - 1 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    p2 = [x - 1 for x in [4, 5, 2, 1, 8, 7, 6, 9, 3]]
    child = [x + 1 for x in order_crossover(p1, p2, 3, 6)]
    assert child == [2, 1, 8, 4, 5, 6, 7, 9, 3]


def test_crossover_dan_mutasi_selalu_permutasi_sah():
    rng = random.Random(1)
    for _ in range(200):
        a = random_chromosome(30, rng.randint(1, 30), rng)
        b = random_chromosome(30, rng.randint(1, 30), rng)
        mask, _ = uniform_crossover(a.mask, b.mask, rng)
        child = Chromosome(mask, order_crossover(a.order, b.order, rng.randrange(10), 10 + rng.randrange(20)))
        mutate(child, 0.5, rng)
        assert sorted(child.order) == list(range(30))
        assert len(set(decode(child))) == len(decode(child))


# ---- loop GA ----


def run(seed):
    return run_ga(problem, GAConfig(max_generations=60, seed=seed))


def test_fitness_terbaik_tidak_pernah_turun_karena_elitisme():
    best = [s.best_fitness for s in run(7).history]
    assert all(b >= a for a, b in zip(best, best[1:]))


def test_seed_sama_hasil_sama():
    assert run(3).best.route == run(3).best.route


def test_menemukan_rencana_valid_untuk_skenario_bawaan():
    assert run(42).best.evaluation.valid
