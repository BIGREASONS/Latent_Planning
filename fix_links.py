with open('manuscript.tex', 'r') as f:
    text = f.read()

text = text.replace(r'\usepackage{hyperref}', r'\usepackage[hidelinks]{hyperref}')

with open('manuscript.tex', 'w') as f:
    f.write(text)
