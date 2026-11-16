"""Generate Bangla humorous and romantic captions with the Gemini API (free tier).

Follows data/style_guide_bn.md, uses the approved pilot + part_0001 as examples, checks every row
with the same rules as tools/check_styled.py, and resumes where it stopped.

Usage (key is read from the GEMINI_API_KEY environment variable):
  python tools/gen_styled_gemini.py --parts 2-72
  python tools/gen_styled_gemini.py --parts 2 --limit 25          # quick test on 25 images
Options: --model gemini-2.5-flash  --batch 25  --rpm 8

Output: data/styled/part_XXXX.tsv.
Work in progress: data/styled/work/part_XXXX.jsonl (one JSON row per image, used for resume).
Stops cleanly when the daily free quota is used up. Run the same command again the next day.
"""
import argparse, collections, csv, json, os, random, re, sys, time, urllib.request, urllib.error

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import check_styled as C   # noqa: E402
import parts as P          # noqa: E402

STYLED = os.path.join(ROOT, 'data', 'styled')
WORK = os.path.join(STYLED, 'work')
GUIDE = open(os.path.join(ROOT, 'data', 'style_guide_bn.md'), encoding='utf-8').read()
FAC = P.captions()


class DailyQuota(Exception):
    pass


class Blocked(Exception):
    pass


class TooLong(Exception):
    pass


BLOCKED = set()   # ids Gemini refuses even alone (safety filter); written to work/blocked.txt


# ---------------------------------------------------------------- examples
def load_examples():
    ex = []
    # prompt examples only (the user-approved pilot style); these rows are NOT part of the dataset
    for f in ('pilot.tsv', 'part_0001.tsv'):
        p = os.path.join(STYLED, 'prompt_examples', f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding='utf-8'), delimiter='\t'):
                if r['image_id'] in FAC:
                    ex.append(r)
    return ex


EXAMPLES = load_examples()


# ---------------------------------------------------------------- row checks (same rules as check_styled.py)
def row_problems(i, h, r):
    refs = FAC[i]
    ref_st = set().union(*[C.stems(c) for c in refs])
    ref_latin = any(re.search(r'[A-Za-z]', c) for c in refs)
    out = []
    for s, c in (('humorous', h), ('romantic', r)):
        n = len(C.words(c))
        if not 8 <= n <= 18: out.append(f'{s}: length {n} words (need 8 to 18)')
        if not c.endswith('।'): out.append(f'{s}: must end with ।')
        if re.search(r'[A-Za-z]', c) and not ref_latin: out.append(f'{s}: English letters')
        if re.search(r'\d{4,}|\.jpg', c): out.append(f'{s}: contains id/number')
        if not (C.stems(c) & ref_st): out.append(f'{s}: not grounded, mention the main subject of the factual captions')
        if max(C.jac(c, x) for x in refs) > 0.75: out.append(f'{s}: too close to a factual caption')
        for t in C.TIME:
            has = lambda text: any(w.startswith(t) for w in C.words(text))
            if has(c) and not any(has(x) for x in refs): out.append(f'{s}: invented time of day "{t}"')
    if C.jac(h, r) > 0.7: out.append('humorous and romantic too similar')
    return out


# ---------------------------------------------------------------- overused phrases (steer away from new templates)
def overused(rows, top=15):
    cnt = collections.Counter()
    for h, r in rows:
        for c in (h, r):
            w = C.words(c)
            for k in range(len(w) - 1):
                cnt[w[k] + ' ' + w[k + 1]] += 1
    n = max(len(rows), 1)
    return [p for p, v in cnt.most_common(60) if v / n > 0.04][:top]


# ---------------------------------------------------------------- Gemini call
def call_gemini(model, key, prompt):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    schema = {'type': 'ARRAY', 'items': {'type': 'OBJECT', 'properties': {
        'id': {'type': 'STRING'}, 'humorous': {'type': 'STRING'}, 'romantic': {'type': 'STRING'}},
        'required': ['id', 'humorous', 'romantic']}}
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.95, 'responseMimeType': 'application/json',
                                 'responseSchema': schema},
            'safetySettings': [{'category': c, 'threshold': 'BLOCK_NONE'} for c in
                               ('HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                                'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT')]}
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.load(resp)
            cand = d.get('candidates') or []
            if not cand or 'content' not in cand[0]:
                why = d.get('promptFeedback', {}).get('blockReason') or (cand[0].get('finishReason') if cand else 'none')
                if why in ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'OTHER', 'RECITATION'):
                    raise Blocked(why)
                raise KeyError(f'no content ({why})')
            txt = cand[0]['content']['parts'][0]['text']
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                if cand[0].get('finishReason') == 'MAX_TOKENS':
                    raise TooLong('reply cut off')   # batch too big: caller splits it, no blind retry
                raise
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', 'ignore')
            if e.code == 429:
                if 'PerDay' in msg or 'per day' in msg.lower():
                    raise DailyQuota(msg[:300])
                m = re.search(r'"retryDelay":\s*"(\d+)', msg)
                wait = int(m.group(1)) + 2 if m else 30 * (attempt + 1)
                print(f'  rate limited, waiting {wait}s', flush=True); time.sleep(wait); continue
            if e.code in (500, 502, 503, 504):
                time.sleep(20 * (attempt + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {msg[:300]}')
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f'  retry after error: {str(e)[:120]}', flush=True); time.sleep(15 * (attempt + 1))
    raise RuntimeError('Gemini failed after retries')


def gen_batch(model, key, items, avoid, fixes):
    """call Gemini; if a batch is blocked, split it in halves until the blocked image is isolated"""
    try:
        return call_gemini(model, key, build_prompt(items, avoid, fixes))
    except TooLong:
        if len(items) == 1: return []
        h = len(items) // 2
        print(f'  reply too long for {len(items)} images, splitting', flush=True)
        return gen_batch(model, key, items[:h], avoid, fixes) + gen_batch(model, key, items[h:], avoid, fixes)
    except Blocked as e:
        if len(items) == 1:
            BLOCKED.add(items[0]); print(f'  blocked by Gemini ({e}): {items[0]}', flush=True)
            with open(os.path.join(WORK, 'blocked.txt'), 'a', encoding='utf-8') as f: f.write(items[0] + chr(10))
            return []
        h = len(items) // 2
        return gen_batch(model, key, items[:h], avoid, fixes) + gen_batch(model, key, items[h:], avoid, fixes)


def build_prompt(items, avoid, fixes=None):
    ex = random.sample(EXAMPLES, min(24, len(EXAMPLES)))
    ex_txt = '\n'.join(json.dumps({'factual': FAC[e['image_id']][:3], 'humorous': e['humorous'],
                                   'romantic': e['romantic']}, ensure_ascii=False) for e in ex)
    inp = []
    for i in items:
        o = {'id': i, 'factual': FAC[i]}
        if fixes and i in fixes: o['problems_to_fix'] = fixes[i]
        inp.append(o)
    return f"""You write one humorous and one romantic Bangla caption for each image, using ONLY its factual captions.
Follow this style guide exactly:

{GUIDE}

Approved examples of the target voice (do not copy them, vary your wording):
{ex_txt}

Phrases already overused in this dataset, avoid them: {', '.join(avoid) if avoid else 'none yet'}

Hard rules: 8 to 18 words each, one sentence ending with "।", keep the main subject and action of the factual
captions, never add a time of day, weather or object that the factual captions do not mention, vary sentence
shape across images, NEVER use "মনে হয়" or "দেখে মনে হয়" in humorous captions, avoid "অদ্ভুত" and
"মূর্ছনা" in romantic captions, humor must mix everyday Bangladeshi life, irony, objects with thoughts and wordplay,
at least 1 in 4 romantic captions address তুমি/তোমার. If an item has problems_to_fix, fix exactly those.

Return a JSON array with one object per input image, same ids, same order.

Input:
{json.dumps(inp, ensure_ascii=False)}"""


# ---------------------------------------------------------------- part driver
def load_work(n):
    p = os.path.join(WORK, f'part_{n:04d}.jsonl')
    done = {}
    if os.path.exists(p):
        for l in open(p, encoding='utf-8'):
            o = json.loads(l); done[o['id']] = o
    return p, done


def batch_limit_offenders(rows):
    """Return (reason, ids) for batch-level limits that are broken, mirroring check_styled.py."""
    N = len(rows); bad = []
    def over(name, test, limit, style):
        ids = [i for i, h, r in rows if test(h if style == 'h' else r)]
        if len(ids) / N > limit:
            random.shuffle(ids); keep = int(limit * N)
            bad.append((name, ids[keep:], test, style))
    over('uses যেন too often', lambda c: 'যেন' in c, .25, 'h')
    over('uses যেন too often', lambda c: 'যেন' in c, .25, 'r')
    over('uses এমন too often', lambda c: 'এমন' in c, .20, 'h')
    over('uses the pattern এমন ... যেন too often', lambda c: bool(re.search(r'এমন.*যেন', c)), .10, 'h')
    over('uses দেখে মনে হয় / মনে হয় too often', lambda c: 'মনে হয়' in c, .15, 'h')
    over('uses অদ্ভুত too often', lambda c: 'অদ্ভুত' in c, .10, 'r')
    over('uses মূর্ছনা too often', lambda c: 'মূর্ছনা' in c, .05, 'r')
    over('uses বুঝি too often', lambda c: 'বুঝি' in c, .15, 'h')
    over('uses ভাবখানা/ভাব দেখে too often', lambda c: bool(re.search(r'ভাবখানা|ভাব দেখে', c)), .10, 'h')
    over('romantic uses a মন-phrase ending too often', lambda c: bool(C.MON.search(c)), .15, 'r')
    over('simile pattern অবিকল/একদম ... মতো/কায়দায় too often', lambda c: bool(C.SIMILE.search(c)), .20, 'h')
    tumi = sum(bool(re.search(r'তুমি|তোমার|তোমাকে', r)) for _, _, r in rows) / N
    if tumi < .20:
        cand = [i for i, h, r in rows if not re.search(r'তুমি|তোমার|তোমাকে', r)]
        random.shuffle(cand)
        bad.append(('romantic should address তুমি/তোমার', cand[:int((.25 - tumi) * N) + 1],
                    lambda c: not re.search(r'তুমি|তোমার|তোমাকে', c), 'r'))
    return bad


def run_part(n, a, key, global_rows):
    ids = P.part_ids(P.ordered_ids(), n)
    if a.limit: ids = ids[:a.limit]
    out_tsv = os.path.join(STYLED, f'part_{n:04d}.tsv')
    if os.path.exists(out_tsv) and not a.limit:
        print(f'part {n}: already finished'); return True
    wp, done = load_work(n)
    # rows saved earlier with problems (accepted after the last try) are retried on every rerun
    fixes = {i: done[i]['problems'] for i in ids if i in done and done[i].get('problems')}
    bf = os.path.join(WORK, 'blocked.txt')
    if os.path.exists(bf): BLOCKED.update(l.strip() for l in open(bf, encoding='utf-8') if l.strip())
    todo = [i for i in ids if (i not in done or i in fixes) and i not in BLOCKED]
    print(f'part {n}: {len(ids) - len(todo)} done, {len(todo)} to write ({len(fixes)} retries)', flush=True)
    wf = open(wp, 'a', encoding='utf-8')
    rounds = 0
    while todo and rounds < 6:
        rounds += 1
        nxt = []
        for s in range(0, len(todo), a.batch):
            items = todo[s:s + a.batch]
            res = gen_batch(a.model, key, items, overused(global_rows), fixes)
            got = {o.get('id'): o for o in res if isinstance(o, dict)}
            for i in items:
                o = got.get(i)
                if not o: nxt.append(i); continue
                h, r = o['humorous'].strip(), o['romantic'].strip()
                probs = row_problems(i, h, r)
                if probs and rounds < 5:
                    fixes[i] = probs; nxt.append(i); continue
                rec = {'id': i, 'humorous': h, 'romantic': r, 'problems': probs, 'model': a.model}
                done[i] = rec; global_rows.append((h, r))
                wf.write(json.dumps(rec, ensure_ascii=False) + '\n'); wf.flush()
            print(f'  part {n}: {len(done)}/{len(ids)}', flush=True)
            time.sleep(60.0 / a.rpm)
        todo = nxt
    # batch-level limits: rewrite a sample of offending rows
    for fix_round in range(a.fix_rounds):
        rows = [(i, done[i]['humorous'], done[i]['romantic']) for i in ids if i in done]
        offenders = batch_limit_offenders(rows)
        if not offenders: break
        redo = collections.defaultdict(list); tests = collections.defaultdict(list)
        for why, oids, test, style in offenders:
            for i in oids: redo[i].append(why + ', rewrite without it'); tests[i].append((test, style))
        print(f'  part {n}: fixing batch-level limits on {len(redo)} rows (round {fix_round + 1})', flush=True)
        rid = list(redo)
        for s in range(0, len(rid), a.batch):
            items = rid[s:s + a.batch]
            res = gen_batch(a.model, key, items, overused(global_rows), redo)
            for o in res:
                i = o.get('id')
                if i not in redo: continue
                # replace only the flagged style, keep the other caption (a new one could break another limit)
                flagged = {st for _, st in tests[i]}
                h2 = o.get('humorous', '').strip() if 'h' in flagged else done[i]['humorous']
                r2 = o.get('romantic', '').strip() if 'r' in flagged else done[i]['romantic']
                if not row_problems(i, h2, r2) and not any(t(h2 if st == 'h' else r2) for t, st in tests[i]):
                    done[i] = {'id': i, 'humorous': h2, 'romantic': r2, 'problems': [], 'model': a.model}
                    wf.write(json.dumps(done[i], ensure_ascii=False) + '\n'); wf.flush()
            time.sleep(60.0 / a.rpm)
    wf.close()
    missing = [i for i in ids if i not in done and i not in BLOCKED]
    if missing:
        print(f'part {n}: {len(missing)} images still missing, rerun to continue'); return False
    target = out_tsv if not a.limit else os.path.join(STYLED, f'test_part_{n:04d}.tsv')
    with open(target, 'w', encoding='utf-8', newline='') as h:
        w = csv.writer(h, delimiter='\t', lineterminator='\n'); w.writerow(['image_id', 'humorous', 'romantic'])
        for i in ids:
            if i in done: w.writerow([i, done[i]['humorous'], done[i]['romantic']])
    print(f'part {n}: written {target}', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='2-72')
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--batch', type=int, default=25)
    ap.add_argument('--rpm', type=float, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fix_rounds', type=int, default=6, help='rounds for batch-level limits (final pass: 12)')
    a = ap.parse_args()
    key = os.environ.get('GEMINI_API_KEY')
    if not key and os.name == 'nt':   # Windows user env var, so the key never goes on a command line
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k: key = winreg.QueryValueEx(k, 'GEMINI_API_KEY')[0]
        except OSError: pass
    if not key: sys.exit('Set GEMINI_API_KEY first.')
    os.makedirs(WORK, exist_ok=True)
    lo, _, hi = a.parts.partition('-'); lo = int(lo); hi = int(hi or lo)
    global_rows = [(e['humorous'], e['romantic']) for e in EXAMPLES]
    for f in sorted(x for x in os.listdir(WORK) if x.endswith('.jsonl')):
        for l in open(os.path.join(WORK, f), encoding='utf-8'):
            o = json.loads(l); global_rows.append((o['humorous'], o['romantic']))
    try:
        for n in range(lo, hi + 1):
            run_part(n, a, key, global_rows)
    except DailyQuota as e:
        print('\nDaily free quota used up. Progress is saved. Run the same command again tomorrow.\n', str(e)[:200])


if __name__ == '__main__':
    main()
