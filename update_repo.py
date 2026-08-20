with open("manuscript.tex", "r") as f:
    text = f.read()

text = text.replace(
    "GitHub: github.com/vaibhav-singh/LatentBench",
    "GitHub: https://github.com/BIGREASONS/Latent\\_Planning",
)

text = text.replace(
    "overcoming earlier caching irregularities.",
    "overcoming earlier caching irregularities. The canonical repository preserving the software and reproducibility artifacts for the experiments reported in this manuscript is available at https://github.com/BIGREASONS/Latent\\_Planning.",
)

with open("manuscript.tex", "w") as f:
    f.write(text)
