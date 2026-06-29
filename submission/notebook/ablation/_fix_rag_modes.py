import json, re

base = "f:/std/GR/code/model_x_ocean/notebook/ablation"
mapping = {
    "rag_ablation_rawpost.ipynb":           "rawpost",
    "rag_ablation_profile.ipynb":           "profile",
    "rag_ablation_facet_embed.ipynb":       "facet_embed",
    "[FIX] rag_ablation_sliced_dual.ipynb": "sliced_dual",
}

for fname, mode in mapping.items():
    path = base + "/" + fname
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        # source can be str or list
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if 'rag_mode = ' not in src or 'hybrid_facet' not in src:
            continue
        # replace the rag_mode assignment line
        new_src = re.sub(
            r'(rag_mode\s*=\s*)"hybrid_facet"([^\n]*)',
            r'\1"' + mode + r'"   # <- change this',
            src,
        )
        if new_src != src:
            if isinstance(cell["source"], list):
                cell["source"] = new_src.splitlines(keepends=True)
            else:
                cell["source"] = new_src
            changed = True
            break

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("Updated " + fname + " -> rag_mode=" + mode)
    else:
        print("WARN: no change in " + fname)
