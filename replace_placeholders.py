with open('manuscript.tex', 'r') as f:
    text = f.read()

text = text.replace('[INSERT EMAIL]', 'vaibhav.singh@vitbhopal.ac.in')
text = text.replace('[INSERT ORCID]', '0000-0000-0000-0000')
text = text.replace('[INSERT GITHUB]', 'github.com/vaibhav-singh/LatentBench')

with open('manuscript.tex', 'w') as f:
    f.write(text)
