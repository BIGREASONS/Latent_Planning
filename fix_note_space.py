with open("manuscript.tex", "r") as f:
    text = f.read()

text = text.replace(
    r"\vspace{1ex}" + "\n" + r"\begin{minipage}{0.95\columnwidth}",
    r"\vspace{2.5ex}" + "\n" + r"\begin{minipage}{0.95\columnwidth}",
)

with open("manuscript.tex", "w") as f:
    f.write(text)
