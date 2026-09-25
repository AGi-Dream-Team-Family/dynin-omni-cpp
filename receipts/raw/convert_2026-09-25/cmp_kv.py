import sys, hashlib
sys.path.insert(0, sys.argv[3])  # gguf-py of the tree
import gguf
def kv(path):
    r = gguf.GGUFReader(path, 'r')
    out = {}
    for name, f in r.fields.items():
        if name.startswith('GGUF.'): continue
        t = f.types[0] if f.types else None
        if t == gguf.GGUFValueType.ARRAY:
            vals = [f.parts[i].tobytes() for i in f.data]
            out[name] = ('ARRAY', f.types[1] if len(f.types) > 1 else None, len(vals), hashlib.sha256(b'\x00'.join(vals)).hexdigest())
        else:
            out[name] = (str(t), f.parts[f.data[0]].tobytes())
    return out
a = kv(sys.argv[1]); b = kv(sys.argv[2])
print("vocab-only keys:", len(a), " reference keys:", len(b))
same = [k for k in a if k in b and a[k] == b[k]]
diff = [k for k in a if k in b and a[k] != b[k]]
only_a = [k for k in a if k not in b]; only_b = [k for k in b if k not in a]
print("identical keys (%d):" % len(same)); [print("  =", k) for k in sorted(same)]
print("differing keys (%d):" % len(diff))
for k in sorted(diff):
    def show(v): return v if v[0]=='ARRAY' else (v[0], v[1][:80])
    print("  !", k, "vocab-only:", show(a[k]), "reference:", show(b[k]))
print("only in vocab-only (%d):" % len(only_a), sorted(only_a))
print("only in reference (%d):" % len(only_b), sorted(only_b))
