# model_selection.py
# Factory for selecting models like xception, mesonet, etc.
# Works with your existing xception.py, models.py, mesonet.py.

from .xception import xception
from .mesonet import Meso4, MesoInception4
from .models import simple_cnn  # if this function does not exist, remove this line

def model_selection(modelname='xception', num_out_classes=2, dropout=0.5):
    """
    Returns a model based on the given architecture name.
    """

    modelname = modelname.lower()

    if modelname == 'xception':
        # Xception model from xception.py
        # Make sure xception() accepts num_classes and dropout.
        return xception(num_classes=num_out_classes, dropout=dropout)

    elif modelname == 'meso4':
        return Meso4(num_classes=num_out_classes)

    elif modelname == 'mesoinc4':
        return MesoInception4(num_classes=num_out_classes)

    elif modelname == 'simple':
        # if simple_cnn is missing in models.py, remove it
        return simple_cnn(num_classes=num_out_classes)

    else:
        raise ValueError(f"Unknown model: {modelname}. Supported: xception, meso4, mesoinc4, simple")
