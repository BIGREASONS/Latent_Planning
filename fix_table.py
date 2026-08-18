with open('manuscript.tex', 'r') as f:
    text = f.read()

# remove clearpage
text = text.replace('\\clearpage\n\\begin{thebibliography}', '\\begin{thebibliography}')

# use float package
if r'\usepackage{float}' not in text:
    text = text.replace(r'\usepackage{booktabs}', r'\usepackage{booktabs}' + '\n' + r'\usepackage{float}')

# force table placement
text = text.replace(r'\begin{table}[h]', r'\begin{table}[H]')

with open('manuscript.tex', 'w') as f:
    f.write(text)
