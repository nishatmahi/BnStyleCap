"""Independent LLM judge from a different model family (OpenAI gpt-oss-120b, open weights, via Groq's free API), same rubric and
output format as tools/judge.py (Gemini), so the two judges can be compared with tools/judge_agree.py.
Following the G-Eval prompting protocol (Liu et al., EMNLP 2023): explicit criteria, integer 1 to 5 scores.
Images: the 212 test images that have human FlickrStyle references. Systems: factual model, Qwen2.5-VL 7B, ours.

  python tools/judge_llama.py --csv "../test_captions_1000_with_qwen.csv" --out ../judge_llama
Key: GROQ_API_KEY (env or Windows user env). Resumable. Free tier: 1K requests and 100K tokens per day,
so a run may stop at the daily cap; rerun the same command after the reset to continue.
"""
import argparse, csv, json, os, re, sys, time, unicodedata, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import parts as P                                   # noqa: E402
from judge import prompt_for, summarize            # same rubric as the Gemini judge  # noqa: E402

API = {'groq': ('https://api.groq.com/openai/v1/chat/completions', 'GROQ_API_KEY', 'openai/gpt-oss-120b'),
       'cerebras': ('https://api.cerebras.ai/v1/chat/completions', 'CEREBRAS_API_KEY', 'gpt-oss-120b')}
URL = API['groq'][0]; KEYNAME = 'GROQ_API_KEY'; EFFORT = 'medium'; USAGE = []
SYSTEMS = [('factual_model', 'humorous', 'factual_generated'), ('factual_model', 'romantic', 'factual_generated'),
           ('qwen25vl_7b', 'humorous', 'qwen_humorous'), ('qwen25vl_7b', 'romantic', 'qwen_romantic'),
           ('ours', 'humorous', 'humorous_generated'), ('ours', 'romantic', 'romantic_generated')]


class DailyCap(Exception):
    pass


def get_key():
    k = os.environ.get(KEYNAME)
    if not k and os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as h: k = winreg.QueryValueEx(h, KEYNAME)[0]
    return k


def call(model, key, prompt):
    body = {'model': model, 'temperature': 0.0, 'response_format': {'type': 'json_object'}, 'reasoning_effort': EFFORT,
            'messages': [{'role': 'user', 'content': prompt + '\n\nReturn ONLY a JSON object of the form '
                          '{"results": [{"id": ..., "style": 1-5, "grounding": 1-5, "fluency": 1-5, '
                          '"creativity": 1-5, "invented": true/false}, ...]}'}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}',
                                          'User-Agent': 'Mozilla/5.0'})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r: d = json.load(r)
            USAGE.append(d.get('usage', {}).get('total_tokens', 0))
            return json.loads(d['choices'][0]['message']['content']).get('results', [])
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', 'ignore')
            if e.code == 429:
                if 'per day' in msg.lower() or 'TPD' in msg or 'RPD' in msg: raise DailyCap(msg[:200])
                m = re.search(r'try again in ([\d.]+)s', msg); time.sleep(float(m.group(1)) + 2 if m else 30); continue
            if e.code >= 500: time.sleep(20 * (attempt + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {msg[:300]}')
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print('  retry:', str(e)[:80], flush=True); time.sleep(15 * (attempt + 1))
    raise RuntimeError('judge call failed')


def main(a):
    global URL, KEYNAME, EFFORT
    EFFORT = a.effort
    URL, KEYNAME, default_model = API[a.api]
    if not a.model: a.model = default_model
    key = get_key()
    if not key: sys.exit(KEYNAME + ' missing')
    os.makedirs(a.out, exist_ok=True)
    fac = P.captions()
    hum = {r['image_id'] for r in csv.DictReader(open(a.human, encoding='utf-8-sig'))}
    rows = [r for r in csv.DictReader(open(a.csv, encoding='utf-8-sig')) if r['image_id'] in hum]
    outp = os.path.join(a.out, 'judge_scores.jsonl')
    done = set()
    if os.path.exists(outp):
        done = {(o['system'], o['style'], o['id']) for o in map(json.loads, open(outp, encoding='utf-8'))}
    f = open(outp, 'a', encoding='utf-8')
    try:
        for sysname, style, col in SYSTEMS:
            todo = [r for r in rows if (sysname, style, r['image_id']) not in done and r[col].strip()]
            print(f'{sysname} {style}: {len(todo)} to judge', flush=True)
            for s in range(0, len(todo), a.batch):
                chunk = todo[s:s + a.batch]
                items = [{'id': r['image_id'], 'factual': fac.get(r['image_id'], [r['factual_reference']])[:a.nfact],
                          'candidate': unicodedata.normalize('NFC', r[col])} for r in chunk]
                got = {str(o.get('id')): o for o in call(a.model, key, prompt_for(style, items)) if isinstance(o, dict)}
                for r in chunk:
                    o = got.get(r['image_id'])
                    try:
                        rec = {'system': sysname, 'style': style, 'id': r['image_id'], 'style_score': int(o['style']),
                               'grounding': int(o['grounding']), 'fluency': int(o['fluency']),
                               'creativity': int(o['creativity']), 'invented': bool(o['invented']), 'model': a.model}
                    except (TypeError, KeyError, ValueError):
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                print(f'  {sysname} {style}: {min(s + a.batch, len(todo))}/{len(todo)}  tokens so far {sum(USAGE)}', flush=True)
                if a.max_batches and len(USAGE) >= a.max_batches: raise DailyCap('test stop')
                time.sleep(a.pause)
    except DailyCap as e:
        print('Daily cap reached, progress saved. Rerun the same command after the reset.', e)
    f.close()
    summarize(outp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--human', default=os.path.join(HERE, '..', '..', 'human_refs', 'human_refs_212_final.csv'))
    ap.add_argument('--out', default='judge_llama')
    ap.add_argument('--api', default='groq', choices=list(API))
    ap.add_argument('--model', default=None, help='default: gpt-oss-120b on the chosen API')
    ap.add_argument('--batch', type=int, default=15)
    ap.add_argument('--effort', default='medium', choices=['low', 'medium', 'high'])
    ap.add_argument('--nfact', type=int, default=3, help='factual captions shown per item')
    ap.add_argument('--max_batches', type=int, default=0, help='stop after N requests (token test)')
    ap.add_argument('--pause', type=float, default=20)
    main(ap.parse_args())
