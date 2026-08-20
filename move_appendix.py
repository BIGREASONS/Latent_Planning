with open("manuscript.tex", "r") as f:
    text = f.read()

bib_start = text.find(r"\begin{thebibliography}")
doc_end = text.find(r"\end{document}")
appendix_start = text.find(r"\appendix")

if bib_start != -1 and appendix_start != -1 and appendix_start > bib_start:
    # Extract sections
    before_bib = text[:bib_start]
    bib_section = text[bib_start:appendix_start]
    appendix_section = text[appendix_start:doc_end]
    end_doc = text[doc_end:]

    # Reorder
    new_text = before_bib + appendix_section + bib_section + end_doc

    with open("manuscript.tex", "w") as f:
        f.write(new_text)
