import json, sys
sys.stdout.reconfigure(encoding="utf-8")
with open("f:/std/GR/code/model_x_ocean/notebook/ablation/rag_ablation_rawpost.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "rag_mode" in src:
            print("source type:", type(cell["source"]))
            print("len source items:", len(cell["source"]))
            # print all lines that contain rag_mode or hybrid_facet
            for i, item in enumerate(cell["source"]):
                if "rag_mode" in item or "hybrid_facet" in item:
                    print(f"[{i}]: {repr(item)}")
            break
