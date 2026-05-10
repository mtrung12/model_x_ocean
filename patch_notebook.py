"""Patch rag_profile_half2_predict.ipynb: replace self._embed() calls
in _embed_query_profile with get_embedding() + np.array wrap."""

import json, sys

nb_path = r"F:\std\GR\code\model_x_ocean\notebook\gpt\rag_profile_half2_predict.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

OLD_FALLBACK = "            return self._embed(query_text)\n"
NEW_FALLBACK  = "            return np.array(get_embedding(query_text), dtype=\"float32\")\n"

OLD_NORMAL   = "        return np.array(self._embed(profile_text), dtype=\"float32\")\n"
NEW_NORMAL    = "        return np.array(get_embedding(profile_text), dtype=\"float32\")\n"

patched = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src_lines = cell.get("source", [])
    new_lines = []
    changed = False
    for line in src_lines:
        if line == OLD_FALLBACK:
            new_lines.append(NEW_FALLBACK)
            changed = True
        elif line == OLD_NORMAL:
            new_lines.append(NEW_NORMAL)
            changed = True
        else:
            new_lines.append(line)
    if changed:
        cell["source"] = new_lines
        patched += 1

if patched == 0:
    print("ERROR: target lines not found – nothing was changed.")
    sys.exit(1)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. {patched} cell(s) patched and saved to:\n  {nb_path}")
