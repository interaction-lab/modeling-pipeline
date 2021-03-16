# Run scripts from the root, as module e.g.
    # python -m examples.test_imports
    
import sys

for p in sys.path:
    print(p)

try:
    import pipeline.common.file_utils as fu
    import pipeline.modeling.data_utils as dat
    import pipeline.annotation.annotate_utterances as au

    print("success")
except ModuleNotFoundError as e:
    raise(e)
