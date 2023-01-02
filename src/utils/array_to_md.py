import numpy as np


def array_to_md(a, decimals: int = 3):
    output = ""
    for row in a:
        output += "| "
        output += " | ".join(map(lambda x: str(round(x, decimals) if not x.is_integer() else int(x)), row))
        output += " |\n"
        
    print(output)
