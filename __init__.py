import logging
import os

from huggingface_hub import snapshot_download

from fiftyone.operators import types

from .zoo import OPERATIONS, MiniCPM_V

MODES = {
    "vqa": "Visual question answering",
    "detection": "Visual question answering",
    "classification": "Visual question answering",
    "keypoint": "Visual question answering"
}

logger = logging.getLogger(__name__)

def download_model(model_name, model_path, **kwargs):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name, model_path, **kwargs):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        **kwargs: optional keyword arguments that configure how the model
            is loaded

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(
            f"Invalid model_path: '{model_path}'. Please ensure the model has been downloaded "
            "using fiftyone.zoo.download_zoo_model(...)"
        )
    
    print(f"Loading MiniCPM-V model from {model_path}")

    # Create and return the model - operations specified at apply time
    return FastVLM(model_path=model_path, **kwargs)


def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
pass