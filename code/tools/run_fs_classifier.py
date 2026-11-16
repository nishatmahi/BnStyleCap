"""Trains the independent style classifier (human FlickrStyle humorous/romantic, Google-translated, plus BanglaView
factual) with style_classifier.py, then scores factual model, Qwen2.5-VL 7B and ours on the 1000 test images.
Runs as a detached process; log in human_refs/fs_clf/clf_log.txt. sklearn hidden (Windows blocks one of its DLLs)."""
import os, sys, csv, json, runpy
sys.modules['sklearn'] = None
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
FS = os.path.join(ROOT, 'human_refs', 'fs_clf'); CLF = os.path.join(FS, 'clf_model'); SC = os.path.join(HERE, '..', 'style_classifier.py')


def run(args):
    sys.argv = ['style_classifier.py'] + args
    try: runpy.run_path(SC, run_name='__main__')
    except SystemExit: pass


if not os.path.exists(os.path.join(CLF, 'config.json')):
    run(['train', '--splits', os.path.join(FS, 'splits'), '--out', CLF, '--epochs', '2'])
rows = list(csv.DictReader(open(os.path.join(ROOT, 'test_captions_1000_with_qwen.csv'), encoding='utf-8-sig')))
for st in ('humorous', 'romantic'):
    for name, col in (('factual_model', 'factual_generated'), ('qwen25vl_7b', f'qwen_{st}'), ('ours', f'{st}_generated')):
        p = os.path.join(FS, f'pred_{name}_{st}.json')
        json.dump([{'image_id': r['image_id'], st: r[col]} for r in rows], open(p, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'=== {name} {st}', flush=True)
        run(['score', '--pred', p, '--style', st, '--out', CLF])
print('ALL DONE', flush=True)
