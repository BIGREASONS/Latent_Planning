with open("manuscript.tex", "r") as f:
    text = f.read()

text = text.replace(r"\begin{thebibliography}", "\\clearpage\n\\begin{thebibliography}")

with open("manuscript.tex", "w") as f:
    f.write(text)
