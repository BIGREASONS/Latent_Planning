with open('manuscript.tex', 'r') as f:
    text = f.read()

# Edit 1: ANOVA phrasing
text = text.replace(
    'indicating a failure to reject the null hypothesis of equal means',
    'which did not provide statistically significant evidence of differences in mean Oracle Gap across architectures'
)

# Edit 2: Probe methodology nuance
text = text.replace(
    'eliminate probe-induced variation.',
    'eliminate probe-induced variation. Probe performance therefore measures recoverability of the evaluated task variables, rather than establishing that the representation is semantically complete.'
)

# Edit 3: Figure 1 caption
text = text.replace(
    'transition architectures fail to produce positive semantic advancement.',
    'the evaluated transition architectures do not produce positive True Action Semantic Gain.'
)

with open('manuscript.tex', 'w') as f:
    f.write(text)
