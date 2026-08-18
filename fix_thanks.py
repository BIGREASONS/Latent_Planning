with open('manuscript.tex', 'r') as f:
    text = f.read()

text = text.replace(r'\thanks{This research received no external funding.}', '')

with open('manuscript.tex', 'w') as f:
    f.write(text)
