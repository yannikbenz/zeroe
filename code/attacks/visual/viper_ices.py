#  Visual attacker based on VAEGAN representations.
#
#  code has been taken from:
#  https://github.com/UKPLab/naacl2019-like-humans-visual-attacks/tree/master/code
#  adaptions were made to integrate it into our workflow
import os
import random

import numpy as np

from gensim.models import KeyedVectors as W2Vec

from code.attacks.visual.perturbations_store import PerturbationsStorage

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
perturbations_file = PerturbationsStorage(os.path.join(SCRIPT_DIR, "perturbations_file"))

repres_file = os.path.join(SCRIPT_DIR, "repres.txt")
print("Read representation file:", repres_file)
model = W2Vec.load_word2vec_format(repres_file)
print("Read representation file done.")


def run(string, prob, top_n=20):
    """
    Perturbs the input string visually in a way that is replaces similar looking chars
    by each other.

    :param string: input string that should be perturbed.
    :param prob: perturbation probability
    :param top_n:
    :return: the visual perturbed string
    """
    isOdd, isEven = False, False
    mydict = {}

    # for line in sys.stdin:
    a = string.split()
    wwords = []
    out_x = []
    for w in a:
        for c in w:
            try:
                if c not in mydict:
                    similar = model.most_similar(c, topn=top_n)
                    if isOdd:
                        similar = [similar[iz] for iz in range(1, len(similar), 2)]
                    elif isEven:
                        similar = [similar[iz] for iz in range(0, len(similar), 2)]
                    words, probs = [x[0] for x in similar], np.array([x[1] for x in similar])
                    probs /= np.sum(probs)
                    mydict[c] = (words, probs)
                else:
                    words, probs = mydict[c]
                r = random.random()
                if r < prob:
                    s = np.random.choice(words, 1, replace=True, p=probs)[0]
                    perturbations_file.add(c, s)
                else:
                    s = c
            except KeyError:
                s = c
            out_x.append(s)
        # out_x.append(" ")
        wwords.append("".join(out_x))
        out_x = []

    perturbations_file.maybe_write()
    return " ".join(wwords)
