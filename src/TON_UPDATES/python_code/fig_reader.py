"""
fig_reader.py - Extract numerical line / errorbar series from MATLAB .fig files.

MATLAB .fig files are MAT-format containers; the figure handle graph (`hgS_*`)
stores each line's XData / YData (and LData / UData for errorbarseries). We
walk that tree and collect every series we find.
"""
import os
import numpy as np
import scipy.io as sio


def _node_props(node):
    return getattr(node, 'properties', None)


def extract_all(fig_path):
    """Return list[dict] - one entry per graphics object that has XData/YData.

    Each entry has keys: type, name, x, y, L, U (errorbar lower / upper deltas).
    """
    d = sio.loadmat(fig_path, struct_as_record=False, squeeze_me=True)
    if 'hgS_070000' not in d:
        raise ValueError(f"{fig_path}: no 'hgS_070000' (not a saved MATLAB fig?)")
    root = d['hgS_070000']

    out = []

    def grab(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.size == 0:
            return None
        return a.astype(float).flatten()

    def walk(node):
        props = _node_props(node)
        if props is not None:
            x = grab(getattr(props, 'XData', None))
            y = grab(getattr(props, 'YData', None))
            if x is not None and y is not None:
                out.append({
                    'type': str(getattr(node, 'type', None) or ''),
                    'name': str(getattr(props, 'DisplayName', '') or '').strip(),
                    'x': x,
                    'y': y,
                    'L': grab(getattr(props, 'LData', None)),
                    'U': grab(getattr(props, 'UData', None)),
                })

        children = getattr(node, 'children', None)
        if children is None:
            return
        if isinstance(children, np.ndarray) and children.dtype == object:
            for c in children.flat:
                walk(c)
        elif hasattr(children, '__iter__') and not isinstance(children, str):
            try:
                for c in children:
                    walk(c)
            except TypeError:
                walk(children)
        else:
            walk(children)

    walk(root)
    return out


def extract_series(fig_path):
    """Compat helper used by older code: dict {display_name: (x, y)}."""
    items = extract_all(fig_path)
    out = {}
    for it in items:
        if it['name'] and it['type'].startswith('graph2d'):
            out[it['name']] = (it['x'], it['y'])
    return out


def extract_named_lines(fig_path):
    """Like extract_series but accepts both graph2d lineseries and any series
    that has a non-empty DisplayName."""
    items = extract_all(fig_path)
    out = {}
    for it in items:
        if it['name']:
            out[it['name']] = (it['x'], it['y'])
    return out


def extract_errorbars(fig_path):
    """Return list of dicts with x, y, L, U for every errorbarseries node."""
    items = extract_all(fig_path)
    return [it for it in items if 'errorbar' in it['type'].lower()]


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for fname in sys.argv[1:] or sorted(f for f in os.listdir(root) if f.endswith('.fig')):
        path = os.path.join(root, fname) if not os.path.isabs(fname) else fname
        if not os.path.isabs(fname) and not os.path.isfile(path):
            path = os.path.join(here, fname)
        print(f"\n=== {fname}")
        try:
            for it in extract_all(path):
                ln = len(it['x'])
                print(f"  type={it['type']}  name={it['name']!r}  n={ln}  "
                      f"hasL={it['L'] is not None}")
        except Exception as e:
            print(f"  ERROR: {e}")
