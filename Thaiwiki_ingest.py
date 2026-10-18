import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from minsearch import Index
from pythainlp.tokenize import word_tokenize  

# ---------------------------------------------------------------- settings
API = "https://th.wikipedia.org/w/api.php"
WIKI_URL = "https://th.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "thai-rag-learning-project/0.2 (personal study project)"}
CACHE_FILE = Path("thaiwiki_documents.jsonl")
DELAY_SECONDS = 1.0     

MAX_CHARS = 800         
MIN_CHARS = 40          
OVERLAP_MAX = 200       
                        


TITLES = [
    "ประเทศไทย",
    "กรุงเทพมหานคร",
    "ประวัติศาสตร์ไทย",
    "ภาษาไทย",
    "อาหารไทย",
    "เชียงใหม่",
    "ภูเก็ต",
    "วัฒนธรรมไทย",
    "เศรษฐกิจไทย",
    "ศาสนาในประเทศไทย",
]

SKIP_SECTIONS = {
    "อ้างอิง", "เชิงอรรถ", "หมายเหตุ", "บรรณานุกรม", "แหล่งข้อมูลอื่น",
    "ดูเพิ่ม", "อ่านเพิ่ม", "ลิงก์ภายนอก", "เว็บไซต์ที่เกี่ยวข้อง",
}

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


# ---------------------------------------------------------------- fetching
def fetch_page(title):
    """Return (real_title, plain_text) for a page. Redirects are followed."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,            
        "exsectionformat": "wiki",   # headings as '== หัวข้อ =='
        "redirects": 1,
        "titles": title,
    }
    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"page not found or empty: {title}")
    return page["title"], page["extract"]


# ---------------------------------------------------------------- parsing
def parse_sections(text):
    """Split page text into (heading_path, body) pairs.

    heading_path is the list of headings above the passage, e.g.
    ['ภูมิศาสตร์', 'ภูมิอากาศ']. The introduction has an empty path.
    """
    sections = []
    path = []  
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(([h for _, h in path], body))

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            level, heading = len(m.group(1)), m.group(2)
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, heading))
        else:
            buf.append(line)
    flush()
    return sections


def split_long(paragraph, max_chars=MAX_CHARS):
    """Cut an overlong paragraph at spaces (Thai sentences are separated by spaces)."""
    pieces = []
    while len(paragraph) > max_chars:
        cut = paragraph.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(paragraph[:cut].strip())
        paragraph = paragraph[cut:].strip()
    if paragraph:
        pieces.append(paragraph)
    return pieces


def chunk_text(body, max_chars=MAX_CHARS, overlap_max=OVERLAP_MAX):
    """Pack paragraphs into passages of at most max_chars characters, with overlap."""
    units = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            units.extend(split_long(line, max_chars))

    chunks, cur, size = [], [], 0
    for unit in units:
        if cur and size + len(unit) + 1 > max_chars:
            chunks.append("\n".join(cur))
            last = cur[-1]
            keep = len(last) <= overlap_max and len(last) + len(unit) + 1 <= max_chars
            cur = [last] if keep else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(unit)
        size += len(unit) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def build_documents(title, text, requested_title=None):
    """Turn one page into passage documents."""
    page_url = WIKI_URL + quote(title.replace(" ", "_"))
    docs, n = [], 0
    for path, body in parse_sections(text):
        if any(h in SKIP_SECTIONS for h in path):
            continue
        section = " > ".join(path) if path else "บทนำ"
        url = page_url + ("#" + quote(path[-1].replace(" ", "_")) if path else "")
        for part in chunk_text(body):
            if len(part) < MIN_CHARS:
                continue
            n += 1
            docs.append(
                {
                    "id": f"{title}#{n}",
                    "title": title,
                    "section": section,
                    "text": part,
                    "url": url,
                    "requested_title": requested_title or title,
                }
            )
    return docs


# ---------------------------------------------------------------- cache
def read_cache():
    if not CACHE_FILE.exists():
        return []
    with CACHE_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_cache(documents):
    tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    tmp.replace(CACHE_FILE)  


def scrape_pages(titles, known_titles=()):
    """Fetch pages one by one. A page that fails is skipped, the others continue."""
    docs, seen = [], set(known_titles)
    for requested in titles:
        try:
            real_title, text = fetch_page(requested)
            if real_title in seen:
                print(f"  {requested}: already have '{real_title}', skipped")
            else:
                page_docs = build_documents(real_title, text, requested_title=requested)
                seen.add(real_title)
                docs.extend(page_docs)
                print(f"  {real_title}: {len(page_docs)} passages")
        except Exception as exc:
            print(f"  SKIPPED {requested}: {exc}", file=sys.stderr)
        time.sleep(DELAY_SECONDS)
    return docs


def load_documents(titles=None, refresh=False):
    """Return all passages. Scrapes only pages that are not cached yet.

    refresh=True scrapes everything again: the pages in `titles` (default TITLES)
    plus any page that was added to the cache earlier.
    """
    wanted = list(titles) if titles else list(TITLES)
    cached = read_cache()

    if refresh:
        extra = [d["requested_title"] for d in cached if d["requested_title"] not in wanted]
        wanted += list(dict.fromkeys(extra))
        cached = []

    have = {d["requested_title"] for d in cached}
    missing = [t for t in wanted if t not in have]
    if missing:
        print(f"Scraping {len(missing)} Thai Wikipedia page(s)...")
        cached += scrape_pages(missing, known_titles={d["title"] for d in cached})
        if cached:
            write_cache(cached)

    if not cached:
        raise RuntimeError("No documents scraped. Check your network connection.")
    return cached


def add_page(title):
    """Scrape one more page into the cache. Returns the number of passages added."""
    cached = read_cache()
    real_title, text = fetch_page(title)
    if any(d["title"] == real_title for d in cached):
        return 0
    docs = build_documents(real_title, text, requested_title=title)
    write_cache(cached + docs)
    return len(docs)


# ---------------------------------------------------------------- indexing
def tokenize_thai(text):
    """Segment Thai text into words and return them as a space-separated string.

    Each word becomes a short hash. minsearch's default tokenizer uses the regex \\w\\w+,
    which cuts Thai words at vowel and tone marks and drops one-character words.
    Hashes are plain alphanumerics, so they always survive. The original text stays
    in the document, so the LLM still gets normal Thai.
    """
    tokens = []
    for w in word_tokenize(text, engine="newmm", keep_whitespace=False):
        if any(ch.isalnum() for ch in w):  
            tokens.append("t" + hashlib.md5(w.lower().encode("utf-8")).hexdigest()[:10])
    return " ".join(tokens)


class ThaiIndex:
    """minsearch index with Thai word segmentation.

    Searchable fields: title, section, text. `title` is also a keyword field, so
    you can restrict a search to one page with filter_dict={"title": "..."}.
    """

    FIELDS = ("title", "section", "text")

    def __init__(self):
        self.index = Index(
            text_fields=[f"{f}_tok" for f in self.FIELDS],
            keyword_fields=["title"],
        )

    def fit(self, documents):
        indexed = []
        for doc in documents:
            item = dict(doc)  
            for f in self.FIELDS:
                item[f"{f}_tok"] = tokenize_thai(doc[f])
            indexed.append(item)
        self.index.fit(indexed)
        return self

    def search(self, query, filter_dict=None, boost_dict=None, num_results=10):
        boost = {f"{k}_tok": v for k, v in (boost_dict or {}).items()}
        return self.index.search(
            tokenize_thai(query),
            filter_dict=filter_dict or {},
            boost_dict=boost,
            num_results=num_results,
        )


def build_index(documents):
    return ThaiIndex().fit(documents)


if __name__ == "__main__":
    docs = load_documents(refresh="--refresh" in sys.argv)
    index = build_index(docs)
    pages = {d["title"] for d in docs}
    print(f"{len(docs)} passages from {len(pages)} pages")
    for hit in index.search("เมืองหลวงของประเทศไทยคืออะไร", num_results=3):
        print(f"- {hit['title']} > {hit['section']}")