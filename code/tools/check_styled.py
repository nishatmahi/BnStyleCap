"""Automatic checks for styled captions (see data/style_guide_bn.md).

Usage:
  python tools/check_styled.py data/styled/part_0001.tsv [more files...]
Each file: TSV with header  image_id  humorous  romantic
Prints a report and writes <file>.fail.tsv with the rows that must be rewritten.
"""
import csv, re, sys, collections, os, unicodedata

FACTUAL = os.path.join(os.path.dirname(__file__), '..', 'data', 'factual_caption.txt')
SIMILE = re.compile(r'(অবিকল|একদম|একেবারে|পুরো|ঠিক|প্রায়|হুবহু)\s+(.{2,40}?)(এর|র|ের)?\s*(মতো|মতন|স্টাইলে|ভঙ্গিতে|কায়দায়)')
FEEL = ['মায়া', 'স্নেহ', 'প্রীতি', 'মুগ্ধতা', 'আবেশ', 'মমতা', 'সোহাগ', 'মোহ', 'আসক্তি', 'অনুরাগ', 'প্রণয়', 'টান', 'পিরিত']
TIME = ['সকাল', 'বিকেল', 'সন্ধ্যা', 'রাত', 'ভোর', 'দুপুর', 'গোধূলি']
MON = re.compile(r'মন(টা)?\s|মনে পড়|মন ছুঁ|মন জুড়|মন ভর')
STOP = set('একটি একজন এবং আর সঙ্গে সাথে দুটি দুই কয়েকটি কয়েকজন তার তাদের একটা এক ও'.split())


def factual():
    cap = collections.defaultdict(list)
    for l in open(FACTUAL, encoding='utf-8'):
        m = re.match(r'^(.+?)\.(jpg|jpeg|png)#\d*\s+(.*)$', l.strip(), re.I)
        if m: cap[m.group(1)].append(m.group(3))
    return cap


def words(s): return re.findall(r'[ঀ-৿]+', unicodedata.normalize('NFC', s))


def stems(s): return {w[:4] for w in words(s) if len(w) > 2 and w not in STOP}


def ngr(s):
    s = ''.join(s.split()); return {s[i:i + 3] for i in range(len(s) - 2)}


def jac(a, b):
    A, B = ngr(a), ngr(b); return len(A & B) / len(A | B) if A and B else 0


def main(files):
    fac = factual()
    rows = []
    for f in files:
        rows += [(f, r) for r in csv.DictReader(open(f, encoding='utf-8'), delimiter='\t')]
    seen = {'humorous': collections.Counter(), 'romantic': collections.Counter()}
    for _, r in rows:
        for s in seen: seen[s][r[s].strip()] += 1
    fails = collections.defaultdict(list); fig = collections.Counter(); sim_n = 0; feel = collections.Counter()
    for f, r in rows:
        i = r['image_id'].strip(); refs = fac.get(i, [])
        ref_st = set().union(*[stems(c) for c in refs]) if refs else set()
        ref_latin = any(re.search(r'[A-Za-z]', c) for c in refs)
        for s in ('humorous', 'romantic'):
            c = r[s].strip(); why = []
            n = len(words(c))
            if not refs: why.append('unknown image id')
            if not 8 <= n <= 18: why.append(f'length {n}')
            if not c.endswith('।'): why.append('no ।')
            if re.search(r'[A-Za-z]', c) and not ref_latin: why.append('latin letters')
            if re.search(r'\d{4,}|\.jpg', c): why.append('id/number')
            if seen[s][c] > 1: why.append('duplicate')
            if refs and not (stems(c) & ref_st): why.append('not grounded')
            if refs and max(jac(c, x) for x in refs) > 0.75: why.append('copy of factual')
            has = lambda text, t: any(w.startswith(t) for w in words(text))  # whole-word start, so সরাতে/বিভোর do not match
            for t in TIME:
                if has(c, t) and not any(has(x, t) for x in refs): why.append(f'invented time {t}')
            if s == 'humorous':
                m = SIMILE.search(c)
                if m: sim_n += 1; fig[m.group(2).strip()] += 1
            else:
                last = words(c)[-1] if words(c) else ''
                for w in FEEL:
                    if last.startswith(w): feel[w] += 1
            if why: fails[(f, i)].append(f'{s}: ' + ', '.join(why))
        if jac(r['humorous'], r['romantic']) > 0.7: fails[(f, i)].append('humorous ~ romantic')
    N = len(rows) or 1
    jen = {s: sum('যেন' in r[s] for _, r in rows) / N for s in ('humorous', 'romantic')}
    emon = sum(bool(re.search(r'এমন.*যেন', r['humorous'])) for _, r in rows) / N
    mon = sum(bool(MON.search(r['romantic'])) for _, r in rows) / N
    tumi = sum(bool(re.search(r'তুমি|তোমার|তোমাকে', r['romantic'])) for _, r in rows) / N
    buji = sum('বুঝি' in r['humorous'] for _, r in rows) / N
    bhab = sum(bool(re.search(r'ভাবখানা|ভাব দেখে', r['humorous'])) for _, r in rows) / N
    emon1 = sum('এমন' in r['humorous'] for _, r in rows) / N
    monehoy = sum('মনে হয়' in r['humorous'] for _, r in rows) / N
    odbhut = sum('অদ্ভুত' in r['romantic'] for _, r in rows) / N
    murchona = sum('মূর্ছনা' in r['romantic'] for _, r in rows) / N
    print(f"humorous মনে হয় {monehoy:.0%} (limit 15%)   romantic অদ্ভুত {odbhut:.0%} (limit 10%)   মূর্ছনা {murchona:.0%} (limit 5%)")
    print(f"বুঝি {buji:.0%} (limit 15%)   ভাবখানা/ভাব দেখে {bhab:.0%} (limit 10%)   এমন {emon1:.0%} (limit 20%)")
    print(f"যেন rate humorous {jen['humorous']:.0%} romantic {jen['romantic']:.0%} (limit 25%)")
    print(f"এমন...যেন rate {emon:.0%} (limit 10%)   romantic মন-phrase rate {mon:.0%} (limit 15%)   তুমি rate {tumi:.0%} (min 20%)")
    bad = [k for k, v in [('যেন', max(jen.values()) > .25), ('এমন-যেন', emon > .10), ('মন', mon > .15), ('তুমি', tumi < .20), ('বুঝি', buji > .15), ('ভাবখানা', bhab > .10), ('এমন', emon1 > .20), ('মনে হয়', monehoy > .15), ('অদ্ভুত', odbhut > .10), ('মূর্ছনা', murchona > .05)] if v]
    print('BATCH-LEVEL LIMITS BROKEN:' if bad else 'batch-level limits OK', bad)
    print(f'rows {len(rows)}  failing rows {len(fails)}')
    print(f'simile rate {sim_n / N:.1%} (limit 20%)')
    print('top simile figures', fig.most_common(5), '(limit 15 each over whole dataset)')
    print('romantic ending feeling words', {k: f'{v / N:.1%}' for k, v in feel.most_common(5)}, '(limit 3% each)')
    for f in files:
        out = f + '.fail.tsv'
        with open(out, 'w', encoding='utf-8', newline='') as h:
            w = csv.writer(h, delimiter='\t'); w.writerow(['image_id', 'reasons'])
            for (ff, i), rs in fails.items():
                if ff == f: w.writerow([i, ' | '.join(rs)])
        print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1:])
