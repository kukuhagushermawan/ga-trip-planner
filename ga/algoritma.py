"""
LANGKAH 4 - LOOP GA
  1. Populasi awal acak
  2. Setiap generasi:
       elitisme   2 individu terbaik disalin utuh
       seleksi    turnamen (k = 3)
       crossover  peluang 0,8: uniform (mask) + order crossover (order)
       mutasi     peluang 0,05 per destinasi: swap, insert, tambah, hapus
  3. Berhenti di generasi maksimum, atau jika fitness terbaik tidak membaik selama 100 generasi
  4. Keluarkan individu terbaik
"""
import random
import time
from dataclasses import dataclass, field

from .fitness import evaluate_route
from .kromosom import Chromosome, clone, decode, random_chromosome
from .operators import mutate, order_crossover_pair, tournament_select, uniform_crossover


@dataclass
class GAConfig:
    population_size: int = 100
    max_generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05  # peluang mutasi per destinasi di rute
    elite_count: int = 2
    tournament_size: int = 3
    stagnation_limit: int = 100  # berhenti jika terbaik tidak membaik selama sekian generasi (0 = nonaktif)
    seed: int = 42


def validate_config(c):
    errors = []
    if c.population_size < 4:
        errors.append("populasi minimal 4")
    if c.max_generations < 1:
        errors.append("generasi minimal 1")
    if not 0 <= c.crossover_rate <= 1:
        errors.append("peluang crossover 0 sampai 1")
    if not 0 <= c.mutation_rate <= 1:
        errors.append("peluang mutasi 0 sampai 1")
    if not 0 <= c.elite_count < c.population_size:
        errors.append("jumlah elit 0 sampai populasi - 1")
    if c.tournament_size < 2:
        errors.append("ukuran turnamen minimal 2")
    if errors:
        raise ValueError("Konfigurasi GA tidak valid: " + ", ".join(errors))


def typical_route_length(problem):
    """Perkiraan jumlah destinasi yang muat dalam batas jam, dipakai sebagai panjang maksimum rute acak."""
    n = problem.n
    visit = sum(problem.visit) / n
    travel = sum(map(sum, problem.travel_h)) / (n * (n - 1))
    return max(1, min(n, round(problem.target_hours / (visit + travel))))


def initial_population(problem, config, rng):
    max_len = typical_route_length(problem)
    return [random_chromosome(problem.n, rng.randint(1, max_len), rng) for _ in range(config.population_size)]


@dataclass
class Individual:
    chromosome: Chromosome
    route: list
    evaluation: object


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    valid_rate: float  # porsi individu valid di populasi
    evaluations: int


@dataclass
class GAResult:
    best: Individual
    history: list = field(default_factory=list)
    generations: int = 0
    evaluations: int = 0
    stop_reason: str = "generasi maksimum"
    seconds: float = 0.0


def run_ga(problem, config=None, on_generation=None):
    config = config or GAConfig()
    validate_config(config)
    rng = random.Random(config.seed)
    started = time.perf_counter()
    history = []
    evaluations = 0

    def evaluate(chromosome):
        nonlocal evaluations
        evaluations += 1
        route = decode(chromosome)
        return Individual(chromosome, route, evaluate_route(problem, route))

    def record(generation, population):
        fitness = [ind.evaluation.fitness for ind in population]
        stats = GenerationStats(
            generation=generation,
            best_fitness=max(fitness),
            mean_fitness=sum(fitness) / len(fitness),
            valid_rate=sum(ind.evaluation.valid for ind in population) / len(population),
            evaluations=evaluations,
        )
        history.append(stats)
        if on_generation:
            on_generation(stats)
        return stats

    def by_fitness(ind):
        return ind.evaluation.fitness

    # 1. Populasi awal
    population = [evaluate(c) for c in initial_population(problem, config, rng)]
    best = record(0, population).best_fitness
    since_improvement = 0
    stop_reason = "generasi maksimum"
    generation = 0

    # 2. Evolusi
    while generation < config.max_generations:
        generation += 1
        fitness = [ind.evaluation.fitness for ind in population]

        def select():
            return population[tournament_select(fitness, config.tournament_size, rng)].chromosome

        # Elitisme: individu terbaik masuk generasi berikutnya tanpa diubah.
        nxt = sorted(population, key=by_fitness, reverse=True)[: config.elite_count]
        while len(nxt) < config.population_size:
            p1, p2 = select(), select()
            if rng.random() < config.crossover_rate:
                m1, m2 = uniform_crossover(p1.mask, p2.mask, rng)
                o1, o2 = order_crossover_pair(p1.order, p2.order, rng)
                children = [Chromosome(m1, o1), Chromosome(m2, o2)]
            else:
                children = [clone(p1), clone(p2)]
            for child in children:
                if len(nxt) >= config.population_size:
                    break
                mutate(child, config.mutation_rate, rng)
                nxt.append(evaluate(child))
        population = nxt

        # 3. Kriteria berhenti
        stats = record(generation, population)
        if stats.best_fitness > best:
            best = stats.best_fitness
            since_improvement = 0
        else:
            since_improvement += 1
            if config.stagnation_limit > 0 and since_improvement >= config.stagnation_limit:
                stop_reason = "stagnasi"
                break

    # 4. Individu terbaik (dijaga elitisme, jadi pasti ada di populasi akhir)
    return GAResult(
        best=max(population, key=by_fitness),
        history=history,
        generations=generation,
        evaluations=evaluations,
        stop_reason=stop_reason,
        seconds=time.perf_counter() - started,
    )
