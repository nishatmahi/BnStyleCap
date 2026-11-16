"""Helper for Phase B: unique image ids in first-appearance order, chunked into parts.

Usage:
  python tools/parts.py info                 # total images, parts, size
  python tools/parts.py ids   <part_number>  # image ids for a part (one per line)
  python tools/parts.py fac   <part_number>  # factual captions for a part (for caption writing)
  python tools/parts.py which <image_id>     # which part an id falls in
"""
import re, sys, os, collections

FACTUAL = os.path.join(os.path.dirname(__file__), '..', 'data', 'factual_caption.txt')
PART_SIZE = 500


def ordered_ids():
    seen = []
    s = set()
    for l in open(FACTUAL, encoding='utf-8'):
        m = re.match(r'^(.+?)\.(jpg|jpeg|png)#\d*\s+(.*)$', l.strip(), re.I)
        if m and m.group(1) not in s:
            s.add(m.group(1)); seen.append(m.group(1))
    return seen


def captions():
    cap = collections.defaultdict(list)
    for l in open(FACTUAL, encoding='utf-8'):
        m = re.match(r'^(.+?)\.(jpg|jpeg|png)#\d*\s+(.*)$', l.strip(), re.I)
        if m:
            cap[m.group(1)].append(m.group(3).strip())
    return cap


def part_ids(ids, n):
    a = (n - 1) * PART_SIZE
    return ids[a:a + PART_SIZE]


def main(argv):
    ids = ordered_ids()
    nparts = (len(ids) + PART_SIZE - 1) // PART_SIZE
    if not argv or argv[0] == 'info':
        print(f'total images {len(ids)}  part size {PART_SIZE}  parts {nparts}')
        print(f'last part has {len(ids) - (nparts - 1) * PART_SIZE} images')
        return
    cmd = argv[0]
    if cmd == 'which':
        i = ids.index(argv[1]); print(f'{argv[1]} -> part {i // PART_SIZE + 1} (row {i % PART_SIZE + 1})')
        return
    n = int(argv[1])
    pid = part_ids(ids, n)
    if cmd == 'ids':
        print('\n'.join(pid))
    elif cmd == 'fac':
        cap = captions()
        for i in pid:
            print(f'@@ {i}\t({len(cap[i])})')
            for c in cap[i]:
                print(f'- {c}')


if __name__ == '__main__':
    main(sys.argv[1:])
