# GA Trip Planner

Rekomendasi rencana perjalanan wisata yang pas dengan anggaran dan waktu, dioptimasi dengan Algoritma Genetika (GA).
Proyek mata kuliah Algoritma Genetika, Universitas Gadjah Mada.

**Status: progress report minggu ke-11 (10 November 2026), progres ±50%.**

## Masalah

Pengguna memasukkan anggaran, lama liburan (hari × jam per hari), jumlah orang, dan minat.
GA memilih kombinasi destinasi (mirip Knapsack) sekaligus urutan kunjungannya (mirip TSP).

Sebuah rencana **valid** jika total biaya ≤ anggaran dan total durasi ≤ hari × jam per hari.

## Progres

| Pekerjaan | Status |
|---|---|
| Formulasi masalah dan studi literatur | Selesai |
| Data dan audit data | Selesai |
| Desain GA: kromosom, fitness, operator | Selesai |
| Prototipe GA dan uji unit | Selesai |
| Perbaikan GA: greedy seeding, mutasi lokal, 5 rencana terbaik | Minggu ke-12 |
| Eksperimen pembanding dan sensitivitas parameter | Minggu ke-12–13 |
| Aplikasi web: peta rute dan jadwal harian | Minggu ke-13 |
| Laporan akhir dan presentasi final | Minggu ke-13–14 |

## Menjalankan

Membutuhkan Python 3.10 atau lebih baru. GA hanya memakai library bawaan Python; `pytest` dipakai untuk uji unit.

```bash
pip install -r requirements.txt
python -m pytest
python main.py
```

Dengan masukan sendiri:

```bash
python main.py --budget 1000000 --days 5 --hours 7 --people 2 --tags museum=1,history=0.5 --seed 42
```

| Opsi | Arti |
|---|---|
| `--budget` `--days` `--hours` `--people` | anggaran (Rp), jumlah hari, jam per hari, jumlah orang |
| `--tags` | bobot minat, mis. `museum=1,history=0.5` |
| `--pop` `--gens` `--pc` `--pm` `--elite` | ukuran populasi, generasi maksimum, peluang crossover, peluang mutasi, jumlah elit |
| `--seed` | seed acak; seed yang sama memberi hasil yang sama |

Tag minat: `outdoors`, `kids-friendly`, `museum`, `art-and-culture`, `history`, `amusement-park`, `beach`, `local-favorite`.

## Struktur

```
data/              dataset asli: 201 destinasi + matriks jarak dan waktu tempuh
ga/
  data.py          0. baca dan validasi data, keluarkan 8 destinasi bermasalah
  kromosom.py      1. representasi kromosom (mask + order)
  fitness.py       2. biaya, durasi, validitas, fitness
  operators.py     3. seleksi, crossover, mutasi
  algoritma.py     4. loop GA: populasi awal, elitisme, kriteria berhenti
main.py            menjalankan GA dari terminal
tests/test_ga.py   uji unit
```

## Desain GA

| Tahap | Isi |
|---|---|
| Kromosom | `mask` (destinasi dipilih) + `order` (urutan kunjungan) |
| Fitness | `1 / (cost_score + duration_score)`, ditambah penalti pelanggaran dan bonus minat |
| Populasi awal | 100 individu acak |
| Seleksi | turnamen, k = 3 |
| Crossover | peluang 0,8: uniform pada `mask` + order crossover pada `order` |
| Mutasi | peluang 0,05 per destinasi: swap, insert, tambah, hapus |
| Elitisme | 2 individu terbaik |
| Berhenti | 200 generasi, atau 100 generasi tanpa perbaikan |

## Data dan lisensi

Dataset dari [GA-pairwise-recommender-system](https://github.com/nnqomariyah/GA-pairwise-recommender-system)
(Halim, Qomariyah & Kazakov), dipakai apa adanya. Dataset tersebut berlisensi GPL-3.0, sehingga proyek ini juga
berlisensi [GPL-3.0](LICENSE).

Referensi: Qomariyah, N. N., & Kazakov, D. (2021). A genetic-based pairwise trip planner recommender system.
*Journal of Big Data, 8*, 77. https://doi.org/10.1186/s40537-021-00470-6

## Tim

- Azhar Maulana (24/533487/PA/22582)
- Kukuh Agus Hermawan (24/533395/PA/22573)
