"""Fetch open-educational content (Wikibooks/Wikipedia, CC BY-SA) into data/documents."""
import os
import re
import sys
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DOCS_DIR = os.path.join("data", "documents")

SOURCES = {
    "data_structures_overview.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Introduction",
        "Wikibooks: Data Structures/Introduction (CC BY-SA)",
    ),
    "arrays.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Arrays",
        "Wikibooks: Data Structures/Arrays (CC BY-SA)",
    ),
    "linked_lists.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/LinkedLists",
        "Wikibooks: Data Structures/LinkedLists (CC BY-SA)",
    ),
    "stacks_queues.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Stacks_and_Queues",
        "Wikibooks: Data Structures/Stacks and Queues (CC BY-SA)",
    ),
    "trees_wikibooks.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Trees",
        "Wikibooks: Data Structures/Trees (CC BY-SA)",
    ),
    "hash_tables.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Hash Tables",
        "Wikibooks: Data Structures/Hash Tables (CC BY-SA)",
    ),
    "graphs.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Graphs",
        "Wikibooks: Data Structures/Graphs (CC BY-SA)",
    ),
    "asymptotic_notation.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Asymptotic Notation",
        "Wikibooks: Data Structures/Asymptotic Notation (CC BY-SA)",
    ),
    "min_max_heaps.txt": (
        "https://en.wikibooks.org/wiki/Data_Structures/Min and Max Heaps",
        "Wikibooks: Data Structures/Min and Max Heaps (CC BY-SA)",
    ),
    "recursion_deep_dive.txt": (
        "https://en.wikipedia.org/wiki/Recursion_(computer_science)",
        "Wikipedia: Recursion (computer science) (CC BY-SA)",
    ),
    "binary_search.txt": (
        "https://en.wikipedia.org/wiki/Binary_search_algorithm",
        "Wikipedia: Binary search algorithm (CC BY-SA)",
    ),
    "dynamic_programming.txt": (
        "https://en.wikipedia.org/wiki/Dynamic_programming",
        "Wikipedia: Dynamic programming (CC BY-SA)",
    ),
    "merge_sort.txt": (
        "https://en.wikipedia.org/wiki/Merge_sort",
        "Wikipedia: Merge sort (CC BY-SA)",
    ),
    "quicksort.txt": (
        "https://en.wikipedia.org/wiki/Quicksort",
        "Wikipedia: Quicksort (CC BY-SA)",
    ),
    "binary_search_trees.txt": (
        "https://en.wikipedia.org/wiki/Binary_search_tree",
        "Wikipedia: Binary search tree (CC BY-SA)",
    ),
    "big_o_notation.txt": (
        "https://en.wikipedia.org/wiki/Big_O_notation",
        "Wikipedia: Big O notation (CC BY-SA)",
    ),
    "dijkstra_algorithm.txt": (
        "https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm",
        "Wikipedia: Dijkstra's algorithm (CC BY-SA)",
    ),
}

NOISE_TOKENS = {
    "navbox", "vertical-navbox", "navigation", "printfooter", "catlinks",
    "hatnote", "infobox", "sidebar", "ambox", "tmbox", "fmbox", "nmbox",
    "toc", "thumb", "reflist", "refbegin", "metadata", "noprint",
    "mw-editsection", "sidebar-list",
}

NOISE_IDS = {"toc", "catlinks", "footer", "mw-navigation", "siteNotice"}


def _is_noise(tag):
    if tag.name in ("html", "body"):
        return False
    classes = tag.get("class") or []
    for cls in classes:
        if cls in NOISE_TOKENS or cls.startswith(("mw-navigation", "vector-toc")):
            return True
    if tag.get("id") in NOISE_IDS:
        return True
    return any(str(cid).startswith("cite_note") for cid in [tag.get("id")])


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    for el in soup.find_all(_is_noise):
        try:
            el.decompose()
        except Exception:
            pass
    main = (
        soup.find("div", id="mw-content-text")
        or soup.find("main")
        or soup.body
        or soup
    )
    text = main.get_text("\n")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"\[\d+\]|\[edit\]|\[citation needed\]|\[hide\]|\[show\]", "", line)
        line = re.sub(r"\s+", " ", line)
        if len(line) > 1:
            lines.append(line)
    out = []
    for i, line in enumerate(lines):
        if i and line == lines[i - 1]:
            continue
        out.append(line)
    return "\n\n".join(out)


def fetch(name, url, credit):
    path = os.path.join(DOCS_DIR, name)
    try:
        resp = requests.get(url, headers={"User-Agent": "AI-Tutor-EduBot/1.0"}, timeout=30)
        resp.raise_for_status()
        text = clean_html(resp.text)
        if len(text) < 2000:
            print(f"SKIP {name}: too short ({len(text)} chars)")
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {credit}\n# Source: {url}\n\n{text}\n")
        print(f"OK   {name}: {len(text)} chars")
        return True
    except Exception as e:
        print(f"FAIL {name}: {e}")
        return False


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    ok = 0
    for name, spec in SOURCES.items():
        if spec is None:
            continue
        if fetch(name, spec[0], spec[1]):
            ok += 1
    print(f"\n{ok}/{len([s for s in SOURCES.values() if s])} documents fetched into {DOCS_DIR}")


if __name__ == "__main__":
    main()
