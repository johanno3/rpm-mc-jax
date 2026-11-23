import sys, traceback, importlib
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
ok = True
for modname in ("test_ewald_reference", "test_mc_reference"):
    m = importlib.import_module(modname)
    for name in sorted(dir(m)):
        if name.startswith("test_"):
            try: getattr(m, name)(); print("PASS", modname, name)
            except Exception: ok = False; print("FAIL", modname, name); traceback.print_exc()
sys.exit(0 if ok else 1)
