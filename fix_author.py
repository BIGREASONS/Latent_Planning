with open("manuscript.tex", "r") as f:
    text = f.read()

text = text.replace(
    "Email: singhvaibhavip@gmail.com \\\\",
    "\\IEEEauthorblockA{Email: singhvaibhavip@gmail.com \\\\",
)

with open("manuscript.tex", "w") as f:
    f.write(text)
