from .dataloading import SeqData
from .models import FCN, CNN, RNN, Hybrid


class Eugene():
    """
    The project class
    """
    def __init__(self, sdatas, models):
        self.sdatas = sdatas
        self.models = models
