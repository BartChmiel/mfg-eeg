# scripts/meta_analysis_v2.py
# Meta-analiza top-k krawędzi z raportów "phase_top_edges_*.txt"
# Ulepszenia vs Twoja wersja:
# 1) Poprawna obsługa "X_Other" + bezpieczne reguły interpretacji (prefixy *_)
# 2) Statystyka: p-value (test dwumianowy) dla powtarzalności k/n
# 3) Dynamika czasowa: profile lagów dla region->region (w całej fazie / per PC)
# 4) Kierunkowość: DI = (A->B - B->A)/(A->B + B->A)
# 5) Więcej diagnostyki: ile linii zparsowano, ile odrzucono, które kanały są poza mapą
# 6) Opcje CLI: katalog wyników, topk-per-subject, scenariusze wbudowane lub "all"

from __future__ import annotations

import os
import re
import glob
import argparse
from math import comb
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


# -----------------------------------------------------------------------------
# Region mapping (możesz rozszerzać)
# -----------------------------------------------------------------------------
REGIONS: Dict[str, List[str]] = {
    "F_Exec": ["Fp1", "Fp2", "Fz"],  # Kora Przedczołowa (Decyzje/Hamowanie)
    "F_Motor": [
        "F3",
        "F4",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "C3",
        "Cz",
        "C4",
    ],  # Kora Ruchowa/Planowanie
    "P_Space": [
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "CP1",
        "CP2",
        "CP5",
        "CP6",
    ],  # Kora Ciemieniowa (Przestrzeń/Czucie)
    "O_Vis": ["PO9", "O1", "Oz", "O2", "PO10"],  # Kora Wzrokowa (Widzenie)
    "T_Temp": ["T7", "T8", "TP9", "TP10"],  # Kora Skroniowa
}


def get_region(channel: str) -> str:
    for region, channels in REGIONS.items():
        if channel in channels:
            return region
    return "X_Other"


# -----------------------------------------------------------------------------
# Interpretacja "najczęstszego" przepływu region->region
# -----------------------------------------------------------------------------
def interpret_flow(region_counts: Counter, total_edges: int) -> Tuple[str, str]:
    if total_edges == 0:
        return "Brak danych", ""

    top_flows = region_counts.most_common(3)
    if not top_flows:
        return "Brak dominacji", ""

    (r1, r2), count = top_flows[0]
    dominance = count / total_edges

    # 1) Feedback wzrokowy: parietal -> occipital
    if r1.startswith("P_") and r2.startswith("O_"):
        return (
            "TOP-DOWN VISUAL FEEDBACK",
            "Kora ciemieniowa (przestrzeń) steruje uwagą wzrokową. Typowe dla precyzji/feedbacku.",
        )

    # 2) Feedforward: occipital -> (parietal lub occipital)
    if r1.startswith("O_") and (r2.startswith("P_") or r2.startswith("O_")):
        return (
            "VISUAL INPUT PROCESSING",
            "Przetwarzanie bodźca wzrokowego i przekazywanie do obszarów asocjacyjnych.",
        )

    # 3) Pętla sensomotoryczna
    if ("Motor" in r1) and (("Motor" in r2) or ("Space" in r2)):
        return (
            "MOTOR COMMAND & SENSORIMOTOR LOOP",
            "Silna interakcja układu ruchowego i czuciowo-przestrzennego.",
        )

    # 4) Wykonawcza kontrola
    if "Exec" in r1:
        return (
            "EXECUTIVE CONTROL (STOP/PLAN)",
            "Kora przedczołowa wysyła sygnały planowania/hamowania do innych obszarów.",
        )

    # 5) Rozproszone
    if dominance < 0.15:
        return (
            "SIEĆ ROZPROSZONA",
            "Brak jednego dominującego przepływu; możliwe duże różnice między badanymi.",
        )

    return f"TRANSMISJA {r1} -> {r2}", "Specyficzne połączenie między regionami."


# -----------------------------------------------------------------------------
# Statystyka i miary kierunku
# -----------------------------------------------------------------------------
def binom_pvalue_ge(k: int, n: int, p0: float = 0.5) -> float:
    """
    P(X >= k) dla Binomial(n, p0).
    Uwaga: to jest test jednostronny "co najmniej tyle sukcesów".
    Dla małych n działa idealnie.
    """
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))
    p0 = float(p0)
    return sum(comb(n, i) * (p0**i) * ((1.0 - p0) ** (n - i)) for i in range(k, n + 1))


def directionality_index(edge_counter: Counter, a: str, b: str) -> float:
    """
    DI = (A->B - B->A)/(A->B + B->A)
    Zakres [-1,1]. 0 = symetria.
    """
    ab = int(edge_counter.get((a, b), 0))
    ba = int(edge_counter.get((b, a), 0))
    denom = ab + ba
    if denom == 0:
        return 0.0
    return (ab - ba) / float(denom)


# -----------------------------------------------------------------------------
# Parsowanie nazw plików i linii z krawędziami
# -----------------------------------------------------------------------------
_FILENAME_RE = re.compile(
    r"""
    ^phase_top_edges_
    (?P<phase>.+?)_
    (?P<mode>gc|corr)_
    m(?P<m>\d+)_               # m4
    pc(?P<pc>\d+)_             # pc1
    lag(?P<lag>\d+)ms          # lag150ms
    \.txt$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_EDGE_LINE_RE = re.compile(
    r"""
    ^\s*(?P<rank>\d+)\.\s*
    (?P<src>[A-Za-z0-9]+)\s*->\s*(?P<dst>[A-Za-z0-9]+)\s*:\s*
    (?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$
    """,
    re.VERBOSE,
)


def parse_filename(
    fname: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    Zwraca (phase, pc, lag_label, pc_int, lag_ms_int) lub (None,...)
    """
    m = _FILENAME_RE.match(fname)
    if not m:
        return None, None, None, None, None

    phase = m.group("phase")
    pc_i = int(m.group("pc"))
    lag_ms = int(m.group("lag"))
    pc = f"pc{pc_i}"
    lag_label = f"lag{lag_ms}ms"
    return phase, pc, lag_label, pc_i, lag_ms


def parse_edge_line(line: str) -> Optional[Tuple[str, str, float]]:
    """
    Parsuje linie typu: "001. PO9 -> O1 : 0.123456"
    Zwraca (src, dst, val) albo None.
    """
    mm = _EDGE_LINE_RE.match(line.strip())
    if not mm:
        return None
    src = mm.group("src").strip()
    dst = mm.group("dst").strip()
    val = float(mm.group("val"))
    return src, dst, val


# -----------------------------------------------------------------------------
# Główna analiza
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default=os.path.join("out", "phase_mats_pca_by_subject"),
        help="Katalog z wynikami (z podkatalogami subjXX i plikami *.txt).",
    )
    ap.add_argument(
        "--topk-per-subject",
        type=int,
        default=5,
        help="Ile krawędzi maksymalnie brać na subjecta dla danego (phase,pc,lag).",
    )
    ap.add_argument(
        "--show-all-scenarios",
        action="store_true",
        help="Jeśli ustawione: wypisze podsumowanie dla wszystkich (phase,pc,lag) znalezionych w plikach.",
    )
    ap.add_argument(
        "--p0",
        type=float,
        default=0.5,
        help="Hipoteza zerowa dla testu dwumianowego (domyślnie 0.5).",
    )
    ap.add_argument(
        "--min-total-edges",
        type=int,
        default=10,
        help="Minimalna liczba zebranych krawędzi, żeby w ogóle raportować scenariusz.",
    )
    ap.add_argument(
        "--temporal-top",
        type=int,
        default=5,
        help="Ile najczęstszych przepływów region->region pokazać w profilu czasowym.",
    )
    args = ap.parse_args()

    results_dir = str(args.results_dir)
    print(f"--- ROZPOCZYNAM META-ANALIZĘ W: {results_dir} ---")

    files = glob.glob(os.path.join(results_dir, "**", "*.txt"), recursive=True)
    if not files:
        print("BŁĄD: Nie znaleziono plików .txt.")
        return

    # db[phase][pc][lag] = {"edges":[{"src","dst","val","subj"}], "subjects":set()}
    db = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"edges": [], "subjects": set()})
        )
    )

    # temporal_region_db[phase][pc][(r_src,r_dst)][lag_ms] += 1
    temporal_region_db = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    total_subjects_found = set()

    parsed_lines = 0
    parsed_edges = 0
    skipped_edge_lines = 0
    skipped_files = 0
    unknown_channels = Counter()

    print(f"Przetwarzanie {len(files)} plików raportów...")

    for fpath in files:
        fname = os.path.basename(fpath)
        phase, pc, lag_label, pc_i, lag_ms = parse_filename(fname)
        if phase is None:
            skipped_files += 1
            continue

        subj_id = os.path.basename(os.path.dirname(fpath))
        total_subjects_found.add(subj_id)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    parsed_lines += 1
                    if "->" not in line or ":" not in line:
                        continue

                    parsed = parse_edge_line(line)
                    if parsed is None:
                        skipped_edge_lines += 1
                        continue

                    src, dst, val = parsed
                    parsed_edges += 1

                    # limit topk-per-subject per (phase,pc,lag)
                    current_len = sum(
                        1
                        for x in db[phase][pc][lag_label]["edges"]
                        if x["subj"] == subj_id
                    )
                    if current_len >= int(args.topk_per_subject):
                        continue

                    db[phase][pc][lag_label]["edges"].append(
                        {"src": src, "dst": dst, "val": val, "subj": subj_id}
                    )
                    db[phase][pc][lag_label]["subjects"].add(subj_id)

                    r_src = get_region(src)
                    r_dst = get_region(dst)
                    if r_src == "X_Other":
                        unknown_channels[src] += 1
                    if r_dst == "X_Other":
                        unknown_channels[dst] += 1

                    # dynamika: zliczamy region->region per lag
                    temporal_region_db[phase][pc][(r_src, r_dst)][lag_ms] += 1

        except Exception:
            # lepiej byłoby logować wyjątek, ale na razie: ciche pominięcie
            continue

    N_SUBJ = len(total_subjects_found)
    print(f"Zidentyfikowano {N_SUBJ} badanych: {sorted(list(total_subjects_found))}")
    print(
        f"Statystyki parsowania: lines={parsed_lines}, edges_parsed={parsed_edges}, "
        f"edge_lines_skipped={skipped_edge_lines}, files_skipped={skipped_files}"
    )
    print("-" * 60)

    if unknown_channels:
        top_unknown = unknown_channels.most_common(10)
        print("Kanały poza mapą REGIONS (top 10):")
        for ch, cnt in top_unknown:
            print(f"  {ch}: {cnt}")
        print("-" * 60)

    # Domyślne scenariusze (Twoje)
    default_scenarios = [
        ("HandStart__FirstDigitTouch", "pc1", "lag0ms", "FAZA 1: INTENCJA RUCHU"),
        ("LiftOff__Replace", "pc1", "lag0ms", "FAZA 2: PODNOSZENIE (SIŁA)"),
        (
            "LiftOff__Replace",
            "pc2",
            "lag50ms",
            "FAZA 2: PODNOSZENIE (ASYMETRIA/MODULACJA)",
        ),
        (
            "Replace__BothReleased",
            "pc3",
            "lag150ms",
            "FAZA 3: PRECYZJA (FEEDBACK 150ms)",
        ),
        (
            "Replace__BothReleased",
            "pc3",
            "lag200ms",
            "FAZA 3: PRECYZJA (FEEDBACK 200ms)",
        ),
    ]

    # Jeśli user chce "all", zbierz wszystkie dostępne kombinacje (phase,pc,lag)
    if args.show_all_scenarios:
        scenarios = []
        for phase in sorted(db.keys()):
            for pc in sorted(db[phase].keys()):
                for lag in sorted(db[phase][pc].keys()):
                    scenarios.append((phase, pc, lag, f"{phase} | {pc} | {lag}"))
    else:
        scenarios = default_scenarios

    for phase, pc, lag_label, title in scenarios:
        print(f"\n>>> {title}")
        print(f"    (Faza: {phase}, {pc}, {lag_label})")

        data = db[phase][pc][lag_label]
        edges = data["edges"]

        if not edges or len(edges) < int(args.min_total_edges):
            print(
                f"    [!] Brak danych / za mało danych (edges={len(edges)} < {args.min_total_edges})."
            )
            continue

        # Counters
        region_counter = Counter()  # (r_src, r_dst) -> count
        detailed_counter = (
            Counter()
        )  # (src, dst) -> count (z topk, więc count <= N_SUBJ*topk)
        detailed_val_sum = defaultdict(float)  # (src,dst) -> suma wartości
        detailed_val_abs_sum = defaultdict(float)

        # do DI w regionach użyjemy region_counter, ale DI po regionach działa sensownie
        for e in edges:
            src = e["src"]
            dst = e["dst"]
            val = float(e["val"])

            r_src = get_region(src)
            r_dst = get_region(dst)

            region_counter[(r_src, r_dst)] += 1
            detailed_counter[(src, dst)] += 1
            detailed_val_sum[(src, dst)] += val
            detailed_val_abs_sum[(src, dst)] += abs(val)

        total_edges = len(edges)

        short_title, description = interpret_flow(region_counter, total_edges)
        print(f"    INTERPRETACJA: [{short_title}]")
        print(f"    OPIS: {description}")

        # Najsilniejsze połączenia: zamiast "most_common", sortuj po (powtarzalność, |średnia wartość|)
        # bo count jest ograniczony topk-per-subject i lepiej wyróżnić stabilne i "mocne" krawędzie.
        candidates = []
        for (src, dst), cnt in detailed_counter.items():
            unique_subjs = len(
                set(e["subj"] for e in edges if e["src"] == src and e["dst"] == dst)
            )
            mean_val = detailed_val_sum[(src, dst)] / float(cnt)
            mean_abs = detailed_val_abs_sum[(src, dst)] / float(cnt)
            candidates.append((unique_subjs, mean_abs, mean_val, src, dst, cnt))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        print("\n    NAJSILNIEJSZE POŁĄCZENIA (Powtarzalność + statystyka):")
        top_conns = candidates[:5]
        for unique_subjs, mean_abs, mean_val, src, dst, cnt in top_conns:
            consistency = (unique_subjs / N_SUBJ) * 100 if N_SUBJ > 0 else 0.0
            bar = "█" * unique_subjs + "░" * (N_SUBJ - unique_subjs)
            pval = binom_pvalue_ge(unique_subjs, N_SUBJ, p0=float(args.p0))
            sign = "+" if mean_val >= 0 else "-"
            print(
                f"    {src:4} -> {dst:4} | {bar} | "
                f"{unique_subjs}/{N_SUBJ} ({consistency:.0f}%) | "
                f"mean={mean_val:.3f} (|mean|={mean_abs:.3f}) [{sign}] | p={pval:.3f}"
            )

        print("\n    GŁÓWNE KANAŁY KOMUNIKACJI (Region -> Region) + DI:")
        top_regions = region_counter.most_common(3)
        for (r1, r2), cnt in top_regions:
            pct = (cnt / total_edges) * 100.0
            di = directionality_index(region_counter, r1, r2)
            print(f"    {r1:8} -> {r2:8} : {pct:5.1f}% aktywności | DI={di:+.2f}")

        # Dodatkowo: udział X_Other w scenariuszu
        other_involved = sum(
            1 for (r1, r2), cnt in region_counter.items() if "X_Other" in (r1, r2)
        )
        if other_involved > 0:
            pct_other = 100.0 * other_involved / float(total_edges)
            print(
                f"\n    [WARN] X_Other bierze udział w {pct_other:.1f}% zliczeń region->region "
                f"(sprawdź mapę REGIONS, jeśli to dużo)."
            )

        print("-" * 40)

    # -----------------------------------------------------------------------------
    # DYNAMIKA CZASOWA: profile lagów dla najczęstszych region->region
    # -----------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DYNAMIKA CZASOWA (Region->Region jako funkcja laga)".upper())
    print("=" * 60)

    # Pokaż dla każdej fazy i PC: top przepływy region->region oraz gdzie (lag_ms) się pojawiają
    for phase in sorted(temporal_region_db.keys()):
        for pc in sorted(temporal_region_db[phase].keys()):
            flow_map = temporal_region_db[phase][
                pc
            ]  # (r_src,r_dst) -> Counter(lag_ms->count)
            # ranking przepływów po sumie
            ranked = sorted(
                flow_map.items(), key=lambda kv: sum(kv[1].values()), reverse=True
            )
            if not ranked:
                continue

            print(f"\nFAZA: {phase} | {pc}")
            shown = 0
            for (r_src, r_dst), lag_counts in ranked:
                total = sum(lag_counts.values())
                if total < 3:
                    continue

                # posortowany profil
                profile = ", ".join(
                    f"{lag}ms={cnt}"
                    for lag, cnt in sorted(lag_counts.items(), key=lambda x: x[0])
                )
                # DI na poziomie regionów: liczymy tylko w tym phase+pc, agregując po lagach (czyli jak często)
                # To jest przybliżenie, ale daje intuicję.
                # DI liczymy z Counter((r_src,r_dst)->count) zbudowanym z sumy po lagach:
                tmp_counter = Counter()
                for (a, b), lc in flow_map.items():
                    tmp_counter[(a, b)] += sum(lc.values())
                di = directionality_index(tmp_counter, r_src, r_dst)

                print(
                    f"  {r_src:8}->{r_dst:8} | total={total:3d} | DI={di:+.2f} | {profile}"
                )
                shown += 1
                if shown >= int(args.temporal_top):
                    break


if __name__ == "__main__":
    main()
