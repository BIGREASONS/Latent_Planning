with open('manuscript.tex', 'r') as f:
    text = f.read()

old_note = r'''\vspace{1ex}
\raggedright
\footnotesize{\textit{Note:} The table compares the evaluation dimensions explicitly used in the cited works; absence of a check mark does not imply that a work cannot support the corresponding analysis.}'''

new_note = r'''\vspace{1ex}
\begin{minipage}{0.95\columnwidth}
\centering
\footnotesize \textit{Note:} The table compares the evaluation dimensions explicitly used in the cited works; absence of a check mark does not imply that a work cannot support the corresponding analysis.
\end{minipage}'''

text = text.replace(old_note, new_note)

with open('manuscript.tex', 'w') as f:
    f.write(text)
