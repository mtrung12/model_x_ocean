"""One-shot patch: replace self._embed( with self._embed_query( in cell 7 of the notebook."""
import json, pathlib, re

nb_path = pathlib.Path(__file__).parent / "notebook" / "gpt" / "rag_profile_half2_predict.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_source = []
    for line in cell["source"]:
        if "self._embed(" in line and "_embed_query_profile" not in line:
            new_line = line.replace("self._embed(", "self._embed_query(")
            print(f"  PATCHED: {line.rstrip()!r} -> {new_line.rstrip()!r}")
            new_source.append(new_line)
            changes += 1
        else:
            new_source.append(line)
    cell["source"] = new_source

if changes == 0:
    print("Nothing to patch — already fixed?")
else:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nDone: {changes} line(s) patched, notebook saved.")
