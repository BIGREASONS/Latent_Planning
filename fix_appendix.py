with open("manuscript.tex", "r") as f:
    text = f.read()

text = text.replace(
    r"\appendix" + "\n" + r"\section*{Final Audited Phase 3 Raw Results}",
    r"\section*{Appendix: Phase 3 Raw Results}",
)

with open("manuscript.tex", "w") as f:
    f.write(text)
