
from tabula import read_pdf
from tabulate import tabulate

#reads table from pdf file
df = read_pdf("abc.pdf", pages="all")  # address of pdf file
print(tabulate(df))
