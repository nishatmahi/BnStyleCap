"""gpt-oss-120b judge, all systems shown side by side (BnStyleCap, Claude, Gemma 3 12B).

For each of the 212 reference-evaluation images the judge sees the factual captions and the three candidate captions
labelled A, B, C in a random order (seeded per image and style), without system names, and scores every candidate
on the same four scales. English words or Latin-script Bangla give 1 on every scale (in the prompt and in code).

  python tools/judge_compare.py            (inputs in data/judge_v2, or JUDGE_DATA; output JUDGE_OUT)
Groq keys: GROQ_API_KEY, GROQ_API_KEY_2 ... from env, Windows user env or Kaggle Secrets; moves to the next key at
the daily cap. Resumable.
"""
import collections, csv, json, os, random, re, sys, time, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge as J, judge_llama as JL, parts as Pa

DATA = os.environ.get('JUDGE_DATA', os.path.join(HERE, '..', 'data', 'judge_v2'))
OUT = os.environ.get('JUDGE_OUT', DATA)
os.makedirs(OUT, exist_ok=True)
JOUT = os.path.join(OUT, 'judge_compare.jsonl')
if not os.path.exists(JOUT) and os.path.exists(os.path.join(DATA, 'judge_compare.jsonl')):   # resume an earlier run
    import shutil; shutil.copy(os.path.join(DATA, 'judge_compare.jsonl'), JOUT)
open(JOUT, 'a').close()
N = lambda t: ' '.join(unicodedata.normalize('NFC', t).split())
OFF = {'humorous': '0.5', 'romantic': '0.4'}
SRC = {'bnstylecap': lambda s: f'{DATA}/gen_{s}_c18k_beam3_off{OFF[s]}.json',
       'claude': lambda s: f'{DATA}/claude_{s}.json',
       'gemma3_12b': lambda s: f'{DATA}/gemma_{s}_1000.json'}
SYS = {name: {s: {x['image_id']: N(x[s]) for x in json.load(open(f(s), encoding='utf-8'))} for s in OFF} for name, f in SRC.items()}
rows = {r['image_id']: r for r in csv.DictReader(open(f'{DATA}/test_captions_1000.csv', encoding='utf-8-sig'))}
hum = [r['image_id'] for r in csv.DictReader(open(f'{DATA}/human_refs_212_final.csv', encoding='utf-8-sig'))]
hids = [i for i in hum if i in rows and all(SYS[n][s].get(i) for n in SYS for s in OFF)]
fac = Pa.captions()
LATIN = re.compile('[A-Za-z]')

NAMES = ['GROQ_API_KEY'] + [f'GROQ_API_KEY_{n}' for n in range(2, 10)]
KEYS = [os.environ[n] for n in NAMES if os.environ.get(n)]
try:
    from kaggle_secrets import UserSecretsClient
    for n in NAMES:
        try:
            v = UserSecretsClient().get_secret(n)
            if v and v not in KEYS: KEYS.append(v)
        except Exception:
            pass
except ImportError:
    pass
if not KEYS: KEYS = [JL.get_key()]
print('Groq keys available:', len(KEYS), '| images', len(hids), flush=True)
JL.EFFORT = 'medium'; MODEL = 'openai/gpt-oss-120b'


def order(s, i):
    names = sorted(SYS); random.Random(f'{s}|{i}').shuffle(names)
    return dict(zip('ABC', names))


def prompt(style, blocks):
    return f"""You are a strict evaluator of Bangla image captions. You cannot see the image. The FACTUAL captions
describe what is really in the image. For each image you get three CANDIDATE captions, A, B and C, written by
different systems. Read all three before scoring, then score EACH candidate on its own. Each candidate should be
{J.STYLE_DEF[style]}

Score each from 1 (very poor) to 5 (excellent):
- style: is the candidate genuinely {style}? A plain description, or a stock template phrase without real {style}
  content, scores 1 or 2.
- grounding: does the candidate fit the factual captions? Wrong scene, or invented people, objects, places or actions,
  lowers the score. Jokes and feelings may go beyond the facts, but the described scene must match.
- fluency: natural, grammatical Bangla written only in Bengali script. The sentence must make sense and every word must
  be used with its correct meaning. A grammatical error, an awkward or unnatural phrase, a word used in a wrong sense,
  or a fragment with no clear meaning lowers the score: 5 only if fully natural, meaningful and error-free, 3 for one
  clear problem, 1 or 2 for a sentence without clear meaning or one a native speaker would not write. Style is not
  part of fluency: a plain but natural sentence can score 5. Emoji are allowed.
- creativity: fresh, specific wording rather than a stock phrase.
- invented: true if the candidate states something concrete (object, person, place, action) that the factual
  captions do not support, otherwise false.
Script rule: English is completely prohibited. If a candidate contains any English word or Bangla written in Latin
letters, give it 1 on style, grounding, fluency and creativity, even if its content matches the factual captions.
Emoji are allowed and do not lower any score.
Be strict and consistent. Return one JSON object per candidate, using the candidate ids given (for example "123:A").

Images:
{json.dumps(blocks, ensure_ascii=False)}"""


def ask(style, ids):
    blocks, want = [], []
    for i in ids:
        o = order(style, i)
        blocks.append({'image': i, 'factual': fac.get(i, [rows[i]['factual_reference']])[:3],
                       'candidates': {f'{i}:{L}': SYS[n][style][i] for L, n in o.items()}})
        want += [f'{i}:{L}' for L in o]
    for attempt in range(2):
        try:
            got = {str(r.get('id')): r for r in JL.call(MODEL, KEY, prompt(style, blocks)) if isinstance(r, dict)}
            if all(w in got for w in want) or len(ids) == 1: return got
            print('  incomplete reply', sum(w in got for w in want), '/', len(want), flush=True)
        except RuntimeError as e:
            print('  batch rejected:', str(e)[:90], flush=True); time.sleep(5)
    if len(ids) == 1: return {}
    got = {}
    for i in ids: got.update(ask(style, [i])); time.sleep(3)
    return got


f = open(JOUT, 'a', encoding='utf-8')
for KEY in KEYS:
    print('using key', KEYS.index(KEY) + 1, 'of', len(KEYS), flush=True)
    done = {(o['style'], o['id']) for o in map(json.loads, open(JOUT, encoding='utf-8')) if o}
    try:
        for s in OFF:
            todo = [i for i in hids if (s, i) not in done]
            print('judge', s, len(todo), 'images to do', flush=True)
            for b in range(0, len(todo), 5):
                chunk = todo[b:b + 5]
                got = ask(s, chunk)
                for i in chunk:
                    o = order(s, i); recs = []
                    for L, name in o.items():
                        r = got.get(f'{i}:{L}')
                        try:
                            sc = {k: int(r[k]) for k in ('style', 'grounding', 'fluency', 'creativity')}
                        except (TypeError, KeyError, ValueError):
                            recs = None; break
                        if LATIN.search(SYS[name][s][i]): sc = dict.fromkeys(sc, 1)   # script rule, also in code
                        recs.append({'system': name, 'position': L, 'style_score': sc['style'], 'grounding': sc['grounding'],
                                     'fluency': sc['fluency'], 'creativity': sc['creativity'], 'invented': bool(r.get('invented'))})
                    if recs:
                        f.write(json.dumps({'style': s, 'id': i, 'model': MODEL, 'scores': recs}, ensure_ascii=False) + '\n'); f.flush()
                print(f'  {s}: {min(b + 5, len(todo))}/{len(todo)}  tokens so far {sum(JL.USAGE)}', flush=True)
                time.sleep(10)
        break
    except JL.DailyCap as e:
        print('DAILY CAP on this key, moving to the next key', str(e)[:120], flush=True)
f.close()

recs = [json.loads(l) for l in open(JOUT, encoding='utf-8') if l.strip()]
print('judged images:', collections.Counter(o['style'] for o in recs), flush=True)
for s in OFF:
    for name in SYS:
        sc = [x for o in recs if o['style'] == s for x in o['scores'] if x['system'] == name]
        m = lambda k: sum(x[k] for x in sc) / max(len(sc), 1)
        print(f"{name:11s} {s:9s} n={len(sc):3d}  style {m('style_score'):.2f}  content {m('grounding'):.2f}  "
              f"fluency {m('fluency'):.2f}  creativity {m('creativity'):.2f}", flush=True)
