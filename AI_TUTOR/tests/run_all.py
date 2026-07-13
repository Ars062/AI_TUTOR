import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

test_modules = [
    "test_config",
    "test_prompt_builder",
    "test_evaluation",
    "test_rag",
    "test_hybrid_retriever",
]

passed = 0
failed = 0

for module_name in test_modules:
    print(f"\n{'='*50}")
    print(f"Running {module_name}...")
    print(f"{'='*50}")
    try:
        __import__(module_name)
        test_mod = sys.modules[module_name]
        test_funcs = [f for f in dir(test_mod) if f.startswith("test_")]
        for func_name in test_funcs:
            try:
                getattr(test_mod, func_name)()
                print(f"  OK {func_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {func_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  FAIL {func_name}: {e}")
                failed += 1
    except Exception as e:
        print(f"  FAIL Could not load module {module_name}: {e}")
        failed += 1

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
