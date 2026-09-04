import pandas as pd  # this reads and works with tabular data (rows and columns)
import ollama  # this connects to the local ai model
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "orders.csv")

# load the csv into a dataframe — think of it as an in-memory spreadsheet
df = pd.read_csv(CSV_PATH)

question = input("Question: ")

# tell the model exactly what columns exist, so it doesn't have to guess column names
columns_info = ", ".join(df.columns)

# instead of asking the model to answer directly, we ask it to write the pandas code
# that would answer it — this is "text-to-pandas", the model translates language into code
prompt = f"""You are given a pandas DataFrame called df with these columns: {columns_info}
Write a single line of pandas code that answers the question below.
Only output the code itself, nothing else — no explanation, no markdown formatting.

Question: {question}
"""

response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": prompt}]
)

code = response["message"]["content"].strip()

# the model sometimes wraps the code in markdown fences even when told not to, so we strip those
code = code.replace("```python", "").replace("```", "").strip()

print("Generated code:", code)

# this actually runs the code the model wrote, against our real dataframe
# note: running model-generated code like this is a real security risk in production systems,
# we're doing it here only because it's our own local data on our own machine
try:
    result = eval(code)
    print("Result:", result)
except Exception as e:
    print("Could not run the generated code:", e)
