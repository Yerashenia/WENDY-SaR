import pandas as pd

df = pd.read_csv("llm-comparator/outputs/results.csv")

df = df.sort_values(
    by=["model", "llm", "query_id"],
    kind="stable"
)

df.to_csv("results_reordered.csv", index=False)