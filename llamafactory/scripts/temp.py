import pandas as pd

df = pd.read_parquet("data/qwen3_tool_cold_start.parquet")

print(df.iloc[0]['conversations'])