"""gpt-oss-120b judge for BnStyleCap trained on BNature. One candidate per image, same rules as judge_compare.py.

  JUDGE_GEN_<style> = generated json, JUDGE_FACT = factual_test.txt, JUDGE_OUT = output folder, JUDGE_N = images per style
Groq keys: GROQ_API_KEY, GROQ_API_KEY_2 ... from Kaggle Secrets, moves to the next key at the daily cap. Resumable.
"""
import collections, json, os, random, re, sys, time, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge as J, judge_llama as JL

STYLES = [s for s in ('humorous', 'romantic') if os.environ.get('JUDGE_GEN_' + s)]
OUT = os.environ['JUDGE_OUT']; os.makedirs(OUT, exist_ok=True)
JOUT = os.path.join(OUT, 'judge_bnature.jsonl'); open(JOUT, 'a').close()
N = lambda t: ' '.join(unicodedata.normalize('NFC', t).split())
CAP = {s: {os.path.splitext(str(x['image_id']))[0]: N(x[s]) for x in json.load(open(os.environ['JUDGE_GEN_' + s], encoding='utf-8'))} for s in STYLES}
fac = collections.defaultdict(list)
for line in open(os.environ['JUDGE_FACT'], encoding='utf-8'):
    if '\t' in line:
        k, c = line.rstrip('\n').split('\t', 1); fac[os.path.splitext(k.split('#')[0])[0]].append(c.strip())
ids = sorted(set(fac) & set.intersection(*[set(CAP[s]) for s in STYLES]))
random.Random(0).shuffle(ids); ids = ids[:int(os.environ.get('JUDGE_N', 880))]
LATIN = re.compile('[A-Za-z]')
NAMES = ['GROQ_API_KEY'] + ['GROQ_API_KEY_%d' % n for n in range(2, 10)]
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
print('Groq keys available:', len(KEYS), '| images', len(ids), flush=True)
assert KEYS, 'no Groq key in Kaggle Secrets'
JL.EFFORT = 'medium'; MODEL = 'openai/gpt-oss-120b'


def prompt(style, blocks):
    return f"""You are a strict evaluator of Bangla image captions. You cannot see the image. The FACTUAL captions
describe what is really in the image. Score each CANDIDATE caption on its own. Each candidate should be
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
Be strict and consistent. Return one JSON object per candidate, using the candidate ids given.

Images:
{json.dumps(blocks, ensure_ascii=False)}"""


def ask(style, chunk):
    blocks = [{'image': i, 'factual': fac[i][:3], 'candidates': {i + ':A': CAP[style][i]}} for i in chunk]
    want = [i + ':A' for i in chunk]
    for _ in range(2):
        try:
            got = {str(r.get('id')): r for r in JL.call(MODEL, KEY, prompt(style, blocks)) if isinstance(r, dict)}
            if all(w in got for w in want) or len(chunk) == 1: return got
        except RuntimeError as e:
            print('  batch rejected:', str(e)[:90], flush=True); time.sleep(5)
    if len(chunk) == 1: return {}
    got = {}
    for i in chunk: got.update(ask(style, [i])); time.sleep(3)
    return got


f = open(JOUT, 'a', encoding='utf-8')
for KEY in KEYS:
    print('using key', KEYS.index(KEY) + 1, 'of', len(KEYS), flush=True)
    done = {(o['style'], o['id']) for o in map(json.loads, open(JOUT, encoding='utf-8'))}
    try:
        for s in STYLES:
            todo = [i for i in ids if (s, i) not in done]
            print('judge', s, len(todo), 'to do', flush=True)
            for b in range(0, len(todo), 5):
                chunk = todo[b:b + 5]; got = ask(s, chunk)
                for i in chunk:
                    r = got.get(i + ':A')
                    try:
                        sc = {k: int(r[k]) for k in ('style', 'grounding', 'fluency', 'creativity')}
                    except (TypeError, KeyError, ValueError):
                        continue
                    if LATIN.search(CAP[s][i]): sc = dict.fromkeys(sc, 1)
                    f.write(json.dumps({'style': s, 'id': i, 'style_score': sc['style'], 'grounding': sc['grounding'], 'fluency': sc['fluency'],
                                        'creativity': sc['creativity'], 'invented': bool(r.get('invented'))}, ensure_ascii=False) + chr(10)); f.flush()
                print(f'  {s}: {min(b + 5, len(todo))}/{len(todo)} tokens {sum(JL.USAGE)}', flush=True)
                time.sleep(4)
        break
    except JL.DailyCap as e:
        print('DAILY CAP, next key', str(e)[:100], flush=True)
f.close()
rec = [json.loads(l) for l in open(JOUT, encoding='utf-8')]
for s in STYLES:
    r = [o for o in rec if o['style'] == s]; m = lambda k: sum(o[k] for o in r) / max(len(r), 1)
    print(f"{s} n={len(r)} style {m('style_score'):.2f} content {m('grounding'):.2f} fluency {m('fluency'):.2f} invented {m('invented'):.2f}", flush=True)
