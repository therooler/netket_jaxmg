import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
# Adjust pattern depending on your filenames
def plot(ns):
    path = f"./outputs/gpu_mem_rank_*_ns_{ns}.csv"
    files = glob.glob(path)

    dfs = []
    if not files:
        print(f"No data found: {path}")
        return
    for f in files:
        # Extract rank and ns from filename
        m = re.search(r"gpu_mem_rank_(\d+)_ns_(\d+)\.csv", f)
        if not m:
            continue
        rank = int(m.group(1))
        ns = int(m.group(2))

        df = pd.read_csv(f, sep=", ")
        df["rank"] = rank
        df["ns"] = ns

        # Clean memory.used

        df["memory.used"] = (
            df["memory.used [MiB]"]
            .str.strip()  # remove leading/trailing spaces like " 1 MiB"
            .str.replace(" MiB", "", regex=False)  # drop the unit
            .astype(int)  # convert to int
        )
        df["memory.total"] = (
            df["memory.total [MiB]"]
            .str.strip()  # remove leading/trailing spaces like " 1 MiB"
            .str.replace(" MiB", "", regex=False)  # drop the unit
            .astype(int)  # convert to int
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dfs.append(df)

    data = pd.concat(dfs)

    # Plot for each ns
    plt.figure(figsize=(12, 6))

    for _, row in data.groupby(["rank", "index"]):
        label = f"rank{row['rank'].iloc[0]}-gpu{row['index'].iloc[0]}"
        plt.plot(row["timestamp"], row["memory.used"], label=label)
    plt.plot(row["timestamp"], row["memory.total"], label="Max GPU", color="black")
    plt.title(f"GPU Memory Usage over Time (ns={ns})")
    plt.xlabel("Time")
    plt.ylabel("Memory Used [MiB]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/memory_usage_ns_{ns}.pdf")
    plt.show()

if __name__ == '__main__':
    for ns in [4096, 8192, 16384, 32768]:
        plot(ns)