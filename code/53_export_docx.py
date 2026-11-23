"""원고 Markdown → HTML → Word(.docx) (2026-11-23). pandoc이 없어 python-markdown과 LibreOffice를 쓴다.

수식($...$)은 유니코드와 아래·위 첨자로 바꾼다(원고에 쓰인 기호만 다룬다). 그림은 base64 data URI로 HTML에
넣어 LibreOffice가 문서 안에 담게 한다. LibreOffice는 샌드박스 밖에서 돌려야 한다(내부 파이프).

사용: python3 53_export_docx.py <원고 .md> <출력 폴더> <soffice.py 경로>
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

SRC, OUT, SOFFICE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)

SYM = [(r"\left|", "|"), (r"\right|", "|"), (r"\mathrm{Bin}", "Bin"), (r"\alpha", "α"), (r"\delta", "δ"),
       (r"\approx", "≈"), (r"\le", "≤"), (r"\ge", "≥"), (r"\in", "∈"), (r"\ln", "ln "), (r"\{", "{"), (r"\}", "}"),
       (r"\,", " "), ("{,}", ","), (r"\%", "%"), (r"\max", "max"), (r"\sum", "Σ")]


def tex(s):
    for a, b in SYM:
        s = s.replace(a, b)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9*])", r"<sup>\1</sup>", s)
    s = s.replace("ln  ", "ln ").replace("ln δ", "ln δ")
    return f"<i>{s.strip()}</i>"


text = SRC.read_text(encoding="utf-8")
text = re.sub(r"\$\$(.+?)\$\$", lambda m: f'<p class="eq">{tex(m.group(1))}</p>', text, flags=re.S)
text = re.sub(r"(?<!\\)\$([^$\n]+?)\$", lambda m: tex(m.group(1)), text)
def img(m):
    b = base64.b64encode((SRC.parent / m.group(2)).resolve().read_bytes()).decode()
    return f'<p class="fig"><img src="data:image/png;base64,{b}" width="620"/></p>'


text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img, text)
html = markdown.markdown(text, extensions=["tables", "sane_lists"])
css = """
body { font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 16pt; } h2 { font-size: 13pt; margin-top: 18pt; } h3 { font-size: 11.5pt; font-style: italic; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 6pt 0; width: 100%; }
th, td { border-top: 0.5pt solid #000; border-bottom: 0.5pt solid #000; padding: 2pt 4pt; vertical-align: top; }
th { font-weight: bold; } p.eq { text-align: center; } p.fig { text-align: center; }
"""
page = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
h = OUT / (SRC.stem + ".html")
h.write_text(page, encoding="utf-8")
subprocess.run([sys.executable, SOFFICE, "--headless", "--convert-to", "docx:MS Word 2007 XML", "--outdir", str(OUT), str(h)],
               check=True)
print("written", OUT / (SRC.stem + ".docx"))
