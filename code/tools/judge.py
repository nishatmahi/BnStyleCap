"""LLM judge for styled captions (BanglaVerse 2026 style, independent of the style classifier).

For every test image the judge sees that image's FACTUAL captions (what is really in the picture) and one
candidate caption, and scores 1 to 5:
  style      : is it genuinely humorous / romantic (a template phrase with no real joke or feeling scores low)
  grounding  : does it fit the facts (invented people, objects, places or actions score low)
  fluency    : natural, grammatical Bangla in Bengali script only (English or Latin-script words lower it)
  script rule: any English word or Latin-script Bangla gives 1 on every scale (also enforced in code); emoji allowed
  creativity : fresh wording, not a stock phrase
plus invented = true/false (mentions something not supported by the factual captions).

Systems are read from the test CSV (one row per test image):
  ours_romantic / ours_humorous      : romantic_generated / humorous_generated
  factual_model_as_romantic/humorous : factual_generated judged as that style (floor for style)
  gemini_ref_romantic / humorous     : romantic_reference / humorous_reference (upper bound)

  python tools/judge.py --csv "../test_captions_1000 (1).csv" --out judge_out
  python tools/judge.py ... --limit 20          # quick test
Key: GEMINI_API_KEY (env or Windows user env). Resumable: rows already judged are skipped.
"""
import argparse, csv, json, os, re, sys, time, collections, unicodedata, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parts as P   # noqa: E402

SYSTEMS = [('ours', 'romantic', 'romantic_generated'), ('ours', 'humorous', 'humorous_generated'),
           ('factual_model', 'romantic', 'factual_generated'), ('factual_model', 'humorous', 'factual_generated'),
           ('gemini_ref', 'romantic', 'romantic_reference'), ('gemini_ref', 'humorous', 'humorous_reference')]
QWEN_SYSTEMS = [('qwen25vl_7b', 'romantic', 'qwen_romantic'), ('qwen25vl_7b', 'humorous', 'qwen_humorous')]
STYLE_DEF = {'humorous': 'HUMOROUS: it should make a Bangla reader smile through a real joke, irony or witty comparison '
                         'that fits this scene.',
             'romantic': 'ROMANTIC: it should express warmth, tenderness, love, longing or emotional beauty that '
                         'fits this scene.'}


class DailyQuota(Exception):
    pass


class Empty(Exception):
    pass


def judge_batch(model, key, style, items):
    """Judge items; if Gemini returns an empty reply, split the batch in halves (a single failing item is skipped)."""
    try:
        return call(model, key, prompt_for(style, items), len(items))
    except Empty:
        if len(items) == 1: print('  skipped', items[0]['id'], flush=True); return []
        h = len(items) // 2
        return judge_batch(model, key, style, items[:h]) + judge_batch(model, key, style, items[h:])


def get_key():
    k = os.environ.get('GEMINI_API_KEY')
    if not k and os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h: k = winreg.QueryValueEx(h, 'GEMINI_API_KEY')[0]
    return k


def call(model, key, prompt, n):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    schema = {'type': 'ARRAY', 'items': {'type': 'OBJECT', 'properties': {
        'id': {'type': 'STRING'}, 'style': {'type': 'INTEGER'}, 'grounding': {'type': 'INTEGER'},
        'fluency': {'type': 'INTEGER'}, 'creativity': {'type': 'INTEGER'}, 'invented': {'type': 'BOOLEAN'}},
        'required': ['id', 'style', 'grounding', 'fluency', 'creativity', 'invented']}}
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.0, 'responseMimeType': 'application/json', 'responseSchema': schema},
            'safetySettings': [{'category': c, 'threshold': 'BLOCK_NONE'} for c in
                               ('HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                                'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT')]}
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r: d = json.load(r)
            cand = d.get('candidates') or []
            if not cand or 'content' not in cand[0]:
                why = (cand[0].get('finishReason') if cand else None) or d.get('promptFeedback')
                print('  empty reply, reason:', why, flush=True)
                raise Empty(str(why))
            return json.loads(cand[0]['content']['parts'][0]['text'])
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', 'ignore')
            if e.code == 429:
                if 'PerDay' in msg: raise DailyQuota(msg[:200])
                m = re.search(r'"retryDelay":\s*"(\d+)', msg); time.sleep(int(m.group(1)) + 2 if m else 30); continue
            if e.code >= 500: time.sleep(20 * (attempt + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {msg[:200]}')
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print('  retry:', str(e)[:80], flush=True); time.sleep(15 * (attempt + 1))
    raise RuntimeError('judge call failed')


def prompt_for(style, items):
    return f"""You are a strict evaluator of Bangla image captions. You cannot see the image. The FACTUAL captions
describe what is really in the image. Judge each CANDIDATE caption, which should be {STYLE_DEF[style]}

Score each from 1 (very poor) to 5 (excellent):
- style: is the candidate genuinely {style}? A plain description, or a stock template phrase (for example
  "... দেখে মনে হয় ..." with no real joke, or a generic emotional ending) without real {style} content, scores 1 or 2.
- grounding: does the candidate fit the factual captions? Wrong scene, or invented people, objects, places or actions,
  lowers the score. Jokes and feelings may go beyond the facts, but the described scene must match.
- fluency: natural, grammatical Bangla written only in Bengali script. English words or any other Latin-script
  text are not fluent Bangla and lower the score. Emoji are allowed. Any grammatical error (wrong verb form, case
  ending, agreement or word order), an awkward or unnatural phrase, a literal translation from English, or a broken
  or incomplete sentence lowers the score: 5 only if fully natural and error-free, 3 for one clear error or awkward
  phrase, 1 or 2 for several errors or a sentence a native speaker would not write.
  The caption must also make sense, and every word must be used with its correct meaning. If a word is used in a
  wrong sense, or the sentence is a fragment with no clear meaning, give fluency 1 or 2 even if the grammar is
  correct.
- creativity: fresh, specific wording rather than a stock phrase.
- invented: true if the candidate states something concrete (object, person, place, action) that the factual
  captions do not support, otherwise false.
Script rule: English is completely prohibited. If the candidate contains any English word or Bangla written in Latin
letters, give it the lowest score, 1, on style, grounding, fluency and creativity, even if its content matches the
factual captions. Emoji are allowed and do not lower any score.
Be strict and consistent. Return one JSON object per candidate, same ids.

Items:
{json.dumps(items, ensure_ascii=False)}"""


def main(a):
    key = get_key()
    if not key: sys.exit('GEMINI_API_KEY missing')
    os.makedirs(a.out, exist_ok=True)
    fac = P.captions()
    rows = list(csv.DictReader(open(a.csv, encoding='utf-8-sig')))
    if a.limit: rows = rows[:a.limit]
    outp = os.path.join(a.out, 'judge_scores.jsonl')
    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding='utf-8'):
            o = json.loads(l); done.add((o['system'], o['style'], o['id']))
    f = open(outp, 'a', encoding='utf-8')
    try:
        for sysname, style, col in (SYSTEMS + QWEN_SYSTEMS if a.with_qwen else SYSTEMS):
            todo = [r for r in rows if (sysname, style, r['image_id']) not in done and r[col].strip()]
            print(f'{sysname} {style}: {len(todo)} to judge', flush=True)
            for s in range(0, len(todo), a.batch):
                chunk = todo[s:s + a.batch]
                items = [{'id': r['image_id'], 'factual': fac.get(r['image_id'], [r['factual_reference']])[:3],
                          'candidate': unicodedata.normalize('NFC', r[col])} for r in chunk]
                res = judge_batch(a.model, key, style, items)
                got = {str(o.get('id')): o for o in res if isinstance(o, dict)}
                for r in chunk:
                    o = got.get(r['image_id'])
                    if not o: continue
                    rec = {'system': sysname, 'style': style, 'id': r['image_id'],
                           **{k: o[k] for k in ('style', 'grounding', 'fluency', 'creativity', 'invented')},
                           'model': a.model}
                    rec['style_score'] = rec.pop('style'); rec['style'] = style
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                print(f'  {sysname} {style}: {min(s + a.batch, len(todo))}/{len(todo)}', flush=True)
                time.sleep(60.0 / a.rpm)
    except DailyQuota as e:
        print('Daily quota used, progress saved. Rerun the same command after reset.', e)
    f.close()
    summarize(outp)


def summarize(path):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path, encoding='utf-8'):
        o = json.loads(l); k = (o['system'], o['style'])
        for m in ('style_score', 'grounding', 'fluency', 'creativity'): agg[k][m].append(o[m])
        agg[k]['invented'].append(1.0 if o['invented'] else 0.0)
    print(f"\n{'system':15s} {'style':9s} {'n':>5s} {'Style':>6s} {'Ground':>7s} {'Fluency':>8s} {'Creat':>6s} {'Invented%':>10s}")
    for (s, st), m in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
        mean = lambda L: sum(L) / len(L)
        print(f"{s:15s} {st:9s} {len(m['grounding']):5d} {mean(m['style_score']):6.2f} {mean(m['grounding']):7.2f} "
              f"{mean(m['fluency']):8.2f} {mean(m['creativity']):6.2f} {100 * mean(m['invented']):9.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='judge_out')
    ap.add_argument('--model', default='gemini-3.5-flash-lite')
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--rpm', type=float, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--with_qwen', action='store_true', help='also judge qwen_romantic / qwen_humorous columns')
    main(ap.parse_args())
