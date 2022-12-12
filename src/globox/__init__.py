from .boundingbox import BoxFormat, Coordinates, BoundingBox
from .annotation import Annotation
from .annotationset import AnnotationSet
from .errors import UnknownImageFormat, ParsingError, FileParsingError
from .evaluation import EvaluationItem, Evaluation, COCOEvaluator