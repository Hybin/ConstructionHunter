categories = {
    "IP": "ip", "NP": "np", "VP": "vp", "PU": "w", "LCP": "fp", "LC": "f", "SYM":  "w",
    "PP": "pp", "CP": "ap", "DNP": "np", "ADVP": "dp", "ADJP": "ap", "QP": "qp", "NN": "n",
    "NR": "n", "NT": "n", "PN": "p", "VV": "v", "VC": "v", "CC": "c", "VE": "v", "VA": "a",
    "AS": "u", "VRD": "v", "CD":  "m", "DT":  "d", "FW":  "n", "IN":  "p", "JJ":  "a",
    "MD":  "u", "PRP":  "r", "RB":  "d", "SP": "y", "CDP": "mp"
}

log_path = "../log/{}.log"

# Consider the max length of the sequence
# The bigger length you set, the more advance GPU the program require
MAX_SEQUENCE_LENGTH = 180

LEARNING_RATE = 5e-5

MIN_LEARNING_RATE = 1e-5
