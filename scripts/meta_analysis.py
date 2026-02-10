import os
import glob
from collections import defaultdict, Counter
import numpy as np

REGIONS = {
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


def get_region(channel):
    for region, channels in REGIONS.items():
        if channel in channels:
            return region
    return "X_Other"


def get_simplified_flow(src, dst):
    """Zamienia F_Exec->O_Vis na F->O dla czytelności."""
    return f"{src.split('_')[0]}->{dst.split('_')[0]}"


def interpret_flow(region_counts, total_edges):

    if total_edges == 0:
        return "Brak danych", ""

    top_flows = region_counts.most_common(3)
    if not top_flows:
        return "Brak dominacji", ""

    (r1, r2), count = top_flows[0]
    dominance = count / total_edges

    # Główne reguły neurobiologiczne
    # 1. Pętla Wzrokowa (Feedback) - Twoje odkrycie w fazie Replace
    if r1.startswith("P_") and r2.startswith("O_"):
        return (
            "TOP-DOWN VISUAL FEEDBACK",
            "Kora ciemieniowa (przestrzeń) steruje uwagą wzrokową. Typowe dla precyzyjnego celowania.",
        )

    # 2. Feedforward (Odbiór bodźca)
    if r1.startswith("O_") and (r2.startswith("P_") or r2.startswith("O_")):
        return (
            "VISUAL INPUT PROCESSING",
            "Przetwarzanie bodźca wzrokowego (Start) i przekazywanie go do kory asocjacyjnej.",
        )

    # 3. Wykonanie Ruchu (Motor Command)
    if "Motor" in r1 and ("Motor" in r2 or "Space" in r2):
        return (
            "MOTOR COMMAND & SENSORIMOTOR LOOP",
            "Silna synchronizacja w korze ruchowej i czuciowej. Generowanie siły i kontrola uścisku.",
        )

    # 4. Kontrola Wykonawcza (Inhibition/Planning)
    if "Exec" in r1:
        return (
            "EXECUTIVE CONTROL (STOP/PLAN)",
            "Kora przedczołowa wysyła instrukcje do innych obszarów. Planowanie ruchu lub sygnał STOP (puszczenie).",
        )

    # 5. Rozproszone
    if dominance < 0.15:
        return (
            "SIEĆ ROZPROSZONA",
            "Brak jednego dominującego kierunku. Mózg pracuje w trybie globalnym lub duże różnice między badanymi.",
        )

    return f"TRANSMISJA {r1} -> {r2}", "Specyficzne połączenie między regionami."


def parse_filename(fname):
    try:
        base = fname.replace(".txt", "")
        if base.startswith("phase_top_edges_"):
            base = base[len("phase_top_edges_") :]

        if "_gc_" in base:
            sep = "_gc_"
        elif "_corr_" in base:
            sep = "_corr_"
        else:
            return None, None, None

        phase_part, rest = base.split(sep)

        parts = rest.split("_")
        pc = next((p for p in parts if p.startswith("pc")), "unknown")
        lag = next((p for p in parts if p.startswith("lag")), "unknown")

        return phase_part, pc, lag
    except:
        return None, None, None


def main():
    RESULTS_DIR = os.path.join(
        "out", "phase_mats_pca_by_subject"
    )  # "article_phase_pca_mats")

    print(f"--- ROZPOCZYNAM META-ANALIZĘ W: {RESULTS_DIR} ---")

    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.txt"), recursive=True)
    if not files:
        print("BŁĄD: Nie znaleziono plików .txt.")
        return

    db = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"edges": [], "subjects": set()})
        )
    )

    total_subjects_found = set()

    print(f"Przetwarzanie {len(files)} plików raportów...")

    for fpath in files:
        phase, pc, lag = parse_filename(os.path.basename(fpath))
        if not phase:
            continue

        subj_id = os.path.basename(os.path.dirname(fpath))
        total_subjects_found.add(subj_id)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if "->" in line and ":" in line:
                        parts = line.split(":")
                        conn = parts[0].split(".")[1].strip()
                        src, dst = [x.strip() for x in conn.split("->")]

                        current_len = len(
                            [
                                x
                                for x in db[phase][pc][lag]["edges"]
                                if x["subj"] == subj_id
                            ]
                        )
                        if current_len < 5:
                            db[phase][pc][lag]["edges"].append(
                                {"src": src, "dst": dst, "subj": subj_id}
                            )
                            db[phase][pc][lag]["subjects"].add(subj_id)
        except Exception:
            continue

    N_SUBJ = len(total_subjects_found)
    print(f"Zidentyfikowano {N_SUBJ} badanych: {sorted(list(total_subjects_found))}")
    print("-" * 60)

    scenarios = [
        # 1. PLANOWANIE
        ("HandStart__FirstDigitTouch", "pc1", "lag0ms", "FAZA 1: INTENCJA RUCHU"),
        # 2. RUCH WŁAŚCIWY (SIŁA)
        ("LiftOff__Replace", "pc1", "lag0ms", "FAZA 2: PODNOSZENIE (SIŁA)"),
        (
            "LiftOff__Replace",
            "pc2",
            "lag50ms",
            "FAZA 2: PODNOSZENIE (ASYMETRIA/MODULACJA)",
        ),
        # 3. ODKŁADANIE (PRECYZJA)
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

    for phase, pc, lag, title in scenarios:
        print(f"\n>>> {title}")
        print(f"    (Faza: {phase}, {pc}, {lag})")

        data = db[phase][pc][lag]
        edges = data["edges"]
        subjects_involved = len(data["subjects"])

        if not edges:
            print("    [!] Brak danych dla tego scenariusza.")
            continue

        region_counter = Counter()
        detailed_counter = Counter()

        for e in edges:
            r_src = get_region(e["src"])
            r_dst = get_region(e["dst"])
            region_counter[(r_src, r_dst)] += 1
            detailed_counter[(e["src"], e["dst"])] += 1

        total_edges = len(edges)

        short_title, description = interpret_flow(region_counter, total_edges)
        print(f"    INTERPRETACJA: [{short_title}]")
        print(f"    OPIS: {description}")

        print("\n    NAJSILNIEJSZE POŁĄCZENIA (Powtarzalność):")
        top_conns = detailed_counter.most_common(5)
        for (src, dst), count in top_conns:
            unique_subjs = len(
                set(e["subj"] for e in edges if e["src"] == src and e["dst"] == dst)
            )
            consistency = (unique_subjs / N_SUBJ) * 100

            bar = "█" * unique_subjs + "░" * (N_SUBJ - unique_subjs)
            print(
                f"    {src:4} -> {dst:4} | {bar} | {unique_subjs}/{N_SUBJ} badanych ({consistency:.0f}%)"
            )

        print("\n    GŁÓWNE KANAŁY KOMUNIKACJI (Region -> Region):")
        for (r1, r2), count in region_counter.most_common(3):
            pct = (count / total_edges) * 100
            print(f"    {r1:8} -> {r2:8} : {pct:.1f}% aktywności")

        print("-" * 40)


if __name__ == "__main__":
    main()
