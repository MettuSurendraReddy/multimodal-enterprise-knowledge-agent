import pymupdf

page = pymupdf.open("NIST.AI.100-1.pdf")[26]
tabs = page.find_tables()
print("tables found:", len(tabs.tables))
for t in tabs.tables:
    print("rows x cols:", t.row_count, "x", t.col_count)
    for row in t.extract()[:6]:
        print(row)
        