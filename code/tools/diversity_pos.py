"""Diversity and word-use tables, following Celona et al. (Neural Comput. Appl. 2025), Tables 4 and 5.

  python tools/diversity_pos.py --csv "test_captions_1000 (1).csv"
  python tools/diversity_pos.py --csv ... --extra qwen_romantic=qwen_romantic.json qwen_humorous=qwen_humorous.json

Diversity (per system, over its 1000 test captions):
  Self-BLEU-4 (mBLEU, lower = more diverse): BLEU-4 of each caption against all other captions of the same system.
  Div-1, Div-2: distinct uni/bigrams divided by the total number of words (Aneja et al. ICCV 2019, ADS-Cap).
POS (per caption, mean +- std): Bangla POS tags from bnlp (BIS tagset) grouped into the categories of Celona Table 5.
Writes diversity_pos.json and diversity_pos.md next to the CSV.
"""
import argparse, csv, json, os, re, unicodedata, collections
import numpy as np

PUNCT = re.compile(r'[।৷॥,.!?;:\'"“”‘’()\[\]{}\-–—…]')
norm = lambda s: ' '.join(PUNCT.sub(' ', unicodedata.normalize('NFC', s)).split())

# BIS tag -> category of Celona Table 5. Punctuation and symbols are not counted. Other tags (e.g. LC)
# count as tokens only.
GROUP = {'NC': 'Noun', 'NP': 'Noun', 'NV': 'Noun', 'NST': 'Noun',
         'JJ': 'Adjective', 'JQ': 'Numeral', 'QC': 'Numeral', 'QF': 'Numeral', 'QO': 'Numeral',
         'RB': 'Adverb', 'AMN': 'Adverb', 'ALC': 'Adverb', 'PP': 'Adposition', 'CCD': 'Conjunction', 'CCS': 'Conjunction', 'CSB': 'Conjunction', 'CCL': 'Conjunction',
         'DAB': 'Determiner', 'DRL': 'Determiner', 'DWH': 'Determiner',
         'PPR': 'Pronoun', 'PRF': 'Pronoun', 'PRL': 'Pronoun', 'PRC': 'Pronoun', 'PWH': 'Pronoun',
         'VM': 'Verb', 'VAUX': 'Verb', 'VA': 'Verb',
         'RPD': 'Particle', 'CX': 'Particle', 'CIN': 'Particle'}
CATS = ['Adjective', 'Adposition', 'Adverb', 'Conjunction', 'Determiner', 'Noun', 'Numeral', 'Particle',
        'Pronoun', 'Verb']


# the bnlp tagger was trained on precomposed য় ড় ঢ়, which NFC splits into letter + nukta
PRECOMP = {'য়': 'য়', 'ড়': 'ড়', 'ঢ়': 'ঢ়'}


def for_tagger(s):
    s = norm(s)
    for k, v in PRECOMP.items(): s = s.replace(k, v)
    return s


def diversity(caps):
    toks = [norm(c).split() for c in caps]
    words = sum(len(t) for t in toks)
    out = {}
    for n in (1, 2):
        grams = {tuple(t[k:k + n]) for t in toks for k in range(len(t) - n + 1)}
        out[f'Div-{n}'] = len(grams) / max(words, 1)
    from pycocoevalcap.bleu.bleu import Bleu
    sel = [' '.join(t) for t in toks]
    gts = {k: sel[:k] + sel[k + 1:] for k in range(len(sel))}
    res = {k: [sel[k]] for k in range(len(sel))}
    out['Self-BLEU-4'] = Bleu(4).compute_score(gts, res, verbose=0)[0][3]
    return out


def pos_stats(caps, tagger):
    rows = []
    for c in caps:
        tags = [t for _, t in tagger.tag(for_tagger(c)) if t not in ('PU', 'RDS', 'RDF', 'RDX', 'SYM')]
        cnt = collections.Counter(GROUP.get(t, 'Other') for t in tags)
        rows.append([len(tags)] + [cnt[k] for k in CATS])
    a = np.array(rows, dtype=float)
    return {k: (a[:, i].mean(), a[:, i].std()) for i, k in enumerate(['Tokens'] + CATS)}


def main(a):
    rows = list(csv.DictReader(open(a.csv, encoding='utf-8-sig')))
    systems = [('Factual model', 'factual_generated'), ('Human factual reference', 'factual_reference'),
               ('Ours romantic', 'romantic_generated'), ('Gemini romantic reference', 'romantic_reference'),
               ('Ours humorous', 'humorous_generated'), ('Gemini humorous reference', 'humorous_reference')]
    caps = {name: [r[col] for r in rows if r[col].strip()] for name, col in systems}
    for spec in a.extra:
        name, path = spec.split('=', 1)
        d = json.load(open(path, encoding='utf-8'))
        caps[name] = [next(v for k, v in x.items() if k != 'image_id') for x in d]
    from bnlp import BengaliPOS
    tagger = BengaliPOS()
    res = {}
    for name, cs in caps.items():
        res[name] = {'n': len(cs), **diversity(cs), 'pos': pos_stats(cs, tagger)}
        print(name, len(cs), {k: round(v, 3) for k, v in res[name].items() if k not in ('pos', 'n')}, flush=True)
    base = os.path.join(os.path.dirname(os.path.abspath(a.csv)), 'diversity_pos')
    json.dump(res, open(base + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    names = list(res)
    md = ['| System | Self-BLEU-4 (lower better) | Div-1 | Div-2 |', '|---|---|---|---|']
    md += [f"| {n} | {res[n]['Self-BLEU-4']:.2f} | {res[n]['Div-1']:.2f} | {res[n]['Div-2']:.2f} |" for n in names]
    md += ['', '| | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 1) + '|']
    for k in ['Tokens'] + CATS:
        md.append(f'| {k} | ' + ' | '.join(f"{res[n]['pos'][k][0]:.2f} ± {res[n]['pos'][k][1]:.2f}" for n in names) + ' |')
    open(base + '.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('wrote', base + '.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--extra', nargs='*', default=[], help='name=path.json, e.g. Qwen romantic=qwen_romantic.json')
    main(p.parse_args())
