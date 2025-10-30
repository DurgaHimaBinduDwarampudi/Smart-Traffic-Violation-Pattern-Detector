import matplotlib.pyplot as plt
import pandas as pd
import os
import ast

def generate_chart():
    csv_file = "violations.csv"
    chart_dir = "static/charts"
    chart_path = os.path.join(chart_dir, "violations_summary.png")

    # Ensure charts directory exists
    os.makedirs(chart_dir, exist_ok=True)

    # If no CSV exists
    if not os.path.exists(csv_file):
        return None, "No Data"

    df = pd.read_csv(csv_file)

    if df.empty or "Violations" not in df.columns:
        return None, "No Data"

    all_violations = []

    # ✅ Handle both string and list-like values properly
    for v in df["Violations"]:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, list):
                        all_violations.extend([x.strip() for x in parsed if x.strip()])
                except:
                    pass
            else:
                all_violations.extend([x.strip() for x in v.split(",") if x.strip()])

    if not all_violations:
        return None, "No Violations"

    counts = pd.Series(all_violations).value_counts()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values)

    plt.title("Traffic Violations Summary", fontsize=14, fontweight='bold')
    plt.xlabel("Violation Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=20, ha="right")

    colors = ["#ff6f61", "#fdd835", "#64b5f6", "#81c784", "#ba68c8"]
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{int(bar.get_height())}", ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    most_common = counts.idxmax() if not counts.empty else "No Violations"
    return chart_path, most_common
