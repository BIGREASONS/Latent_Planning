with open("manuscript.tex", "r") as f:
    text = f.read()

old_text = r"The final reproducibility package, including \texttt{scripts/analysis.py} and \texttt{scripts/reproduce\_phase3.py}, guarantees auditable re-execution from the raw Phase 3 output logs, overcoming earlier caching irregularities."
new_text = r"The final reproducibility package guarantees auditable re-execution of the TinyLlama Phase 3 experiments from the raw output logs, ensuring robust and transparent validation."

text = text.replace(old_text, new_text)

with open("manuscript.tex", "w") as f:
    f.write(text)
