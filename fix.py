with open('manuscript.tex', 'r') as f:
    text = f.read()

text = text.replace('\ = h_{\\\\text{task}} + h_{\\\\text{nuisance}}\\$', '$h = h_{\\text{task}} + h_{\\text{nuisance}}$')
text = text.replace('preserving \\{\\\\text{nuisance}}\\$', 'preserving $h_{\\text{nuisance}}$')
text = text.replace('reconstruction of \\{\\\\text{task}}\\$', 'reconstruction of $h_{\\text{task}}$')

text = text.replace(' = h_{\\text{task}} + h_{\\text{nuisance}}$', '$h = h_{\\text{task}} + h_{\\text{nuisance}}$')
text = text.replace('preserving {\\text{nuisance}}$', 'preserving $h_{\\text{nuisance}}$')
text = text.replace('reconstruction of {\\text{task}}$', 'reconstruction of $h_{\\text{task}}$')

with open('manuscript.tex', 'w') as f:
    f.write(text)
