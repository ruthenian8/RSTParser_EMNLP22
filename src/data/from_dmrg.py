import sys
import os
import json
from .tree import AttachTree

PFORMAT_MARGIN = 1000000000


class DMRGTree(AttachTree):
    def __init__(self, label: str, children: list):
        if label == "EDU" or label == "text":
            label = "text"
        else:
            nuc, rel = label.split("-", maxsplit=1)
            if nuc == "SN":
                nuc = "satellite-nucleus"
            elif nuc == "NS":
                nuc = "nucleus-satellite"
            elif nuc == "NN":
                nuc = "nucleus-nucleus"
            else:
                super().__init__(label, children)
                return
            label = f"{nuc}:{rel}"
        super().__init__(label, children)


def process_file(filename):
    dmrg_filename = filename + ".dmrg"
    edu_filename = filename + ".edus"

    edus = []
    with open(edu_filename, "r") as f:
        for line in f:
            edus.append(line.strip())
    
    with open(dmrg_filename, "r") as f:
        text = f.read()
        tree = DMRGTree.fromstring(text)
        for pos in tree.treepositions('leaves'):
            tree[pos] = str(int(tree[pos]) - 1) # decrement EDU indices by 1
        rst_tree = DMRGTree.convert_to_rst(tree)
        formatted_tree = rst_tree.pformat(margin=PFORMAT_MARGIN)

    data = {
        "path_basename": os.path.basename(filename),
        "doc_id": os.path.basename(filename),
        "rst_tree": formatted_tree,
        "binarised_rst_tree": formatted_tree,
        "attach_tree": tree.pformat(margin=PFORMAT_MARGIN),
        "edu_strings": edus,
        "edu_starts_sentence": [False for i in edus],
        "edu_starts_paragraph": [True] + [False for i in edus[1:]],
    }    
    return data


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.data.from_dmrg <filename>")
        sys.exit(1)
    if os.path.exists(sys.argv[1] + ".dmrg") and os.path.exists(sys.argv[1] + ".edus"):
        data = process_file(sys.argv[1])
        sys.stdout.write(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        raise FileNotFoundError(f"File {sys.argv[1]} not found.")


if __name__ == "__main__":
    main()