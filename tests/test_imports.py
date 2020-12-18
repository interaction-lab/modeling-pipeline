import pipeline.file_utils as fu
import pipeline.modeling.data_utils as dat
import pipeline.modeling.windowing.get_args
import pipeline.annotation.annotate_utterances as au

au.PythonTurnAnnotator

dat.data_loader()

fu.main()
