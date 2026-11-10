"""
Menjalankan GA sekali dari terminal.
  python main.py
  python main.py --budget 1000000 --days 5 --hours 7 --people 2 --tags museum=1,history=0.5 --seed 42
Opsi GA: --pop --gens --pc --pm --elite
"""
import argparse

from ga.algoritma import GAConfig, run_ga
from ga.data import load_dataset
from ga.fitness import TripInput, build_problem


def parse_tags(text):
    weights = {}
    for pair in filter(None, text.split(",")):
        tag, _, w = pair.partition("=")
        weights[tag.strip()] = float(w) if w else 1.0
    return weights


def rp(x):
    return "Rp" + f"{round(x):,}".replace(",", ".")


def main():
    ap = argparse.ArgumentParser(description="Rekomendasi rencana wisata dengan Algoritma Genetika")
    ap.add_argument("--budget", type=int, default=1_000_000, help="anggaran total (Rp)")
    ap.add_argument("--days", type=int, default=5, help="jumlah hari")
    ap.add_argument("--hours", type=float, default=7, help="jam per hari")
    ap.add_argument("--people", type=int, default=2, help="jumlah orang")
    ap.add_argument("--tags", default="", help="bobot minat, mis. museum=1,history=0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=100, help="ukuran populasi")
    ap.add_argument("--gens", type=int, default=200, help="generasi maksimum")
    ap.add_argument("--pc", type=float, default=0.8, help="peluang crossover")
    ap.add_argument("--pm", type=float, default=0.05, help="peluang mutasi")
    ap.add_argument("--elite", type=int, default=2, help="jumlah elit")
    a = ap.parse_args()

    trip = TripInput(a.budget, a.days, a.hours, a.people, parse_tags(a.tags))
    problem = build_problem(load_dataset(), trip)
    config = GAConfig(a.pop, a.gens, a.pc, a.pm, a.elite, seed=a.seed)
    print(f"Anggaran {rp(trip.budget)} | {trip.days} hari x {trip.hours_per_day:g} jam | {trip.people} orang | {problem.n} kandidat")

    def log(s):
        if s.generation % 25 == 0:
            print(f"gen {s.generation:3d} | terbaik {s.best_fitness:.3f} | rata-rata {s.mean_fitness:.3f} | valid {s.valid_rate:.0%}")

    result = run_ga(problem, config, log)
    print(f"Berhenti ({result.stop_reason}) setelah {result.generations} generasi, {result.evaluations} evaluasi, {result.seconds:.1f} detik\n")

    route, e = result.best.route, result.best.evaluation
    print(f"Rencana terbaik: {'VALID' if e.valid else 'TIDAK VALID'} | fitness {e.fitness:.3f}")
    print(
        f"  {len(route)} destinasi | biaya {rp(e.total_cost)} (sisa {rp(trip.budget - e.total_cost)}) | "
        f"{e.total_hours:.1f} jam dari {problem.target_hours:g} jam | {e.distance_km:.1f} km | minat {e.preference:.0%}"
    )
    for k, c in enumerate(route, 1):
        d = problem.destinasi[problem.candidates[c]]
        print(f"  {k:2d}. {d.id} {d.name} ({rp(d.ticket)}/orang, {d.visit_hours:g} jam)")


if __name__ == "__main__":
    main()
