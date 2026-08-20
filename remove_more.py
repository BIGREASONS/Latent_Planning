with open("manuscript.tex", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "School of Computer Science Engineering" in line or "GitHub:" in line:
        continue
    new_lines.append(line)

with open("manuscript.tex", "w") as f:
    f.writelines(new_lines)
