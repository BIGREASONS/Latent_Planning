with open("manuscript.tex", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "VIT Bhopal University" in line or "Madhya Pradesh" in line:
        continue
    new_lines.append(line)

with open("manuscript.tex", "w") as f:
    f.writelines(new_lines)
