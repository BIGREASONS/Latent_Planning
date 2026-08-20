import sys

with open("manuscript.tex", "r") as f:
    text = f.read()

# Fix Email
text = text.replace("vaibhav.singh@vitbhopal.ac.in", "singhvaibhavip@gmail.com")

# Add hyperref
if r"\usepackage{hyperref}" not in text:
    text = text.replace(
        r"\usepackage{booktabs}",
        r"\usepackage{booktabs}" + "\n" + r"\usepackage{hyperref}",
    )

# Fix URLs
text = text.replace(
    r"https://github.com/BIGREASONS/Latent\_Planning",
    r"\url{https://github.com/BIGREASONS/Latent_Planning}",
)

with open("manuscript.tex", "w") as f:
    f.write(text)
