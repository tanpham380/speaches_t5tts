from fastapi import (
    APIRouter,
    Response,
)
import huggingface_hub
from huggingface_hub.hf_api import RepositoryNotFoundError

import hf_utils 
from dependencies import ModelManagerDependency
from model_aliases import ModelId
router = APIRouter()


@router.get("/health", tags=["diagnostic"])
def health() -> Response:
    return Response(status_code=200, content="OK")




@router.get("/api/ps", tags=["experimental"], summary="Get a list of loaded models.")
def get_running_models(
    model_manager: ModelManagerDependency,
) -> dict[str, list[str]]:
    return {"models": list(model_manager.loaded_models.keys())}


@router.post("/api/ps/{model_id:path}", tags=["experimental"], summary="Load a model into memory.")
def load_model_route(model_manager: ModelManagerDependency, model_id: ModelId) -> Response:
    if model_id in model_manager.loaded_models:
        return Response(status_code=409, content="Model already loaded")
    with model_manager.load_model(model_id):
        pass
    return Response(status_code=201)


@router.delete("/api/ps/{model_id:path}", tags=["experimental"], summary="Unload a model from memory.")
def stop_running_model(model_manager: ModelManagerDependency, model_id: str) -> Response:
    try:
        model_manager.unload_model(model_id)
        return Response(status_code=204)
    except (KeyError, ValueError) as e:
        match e:
            case KeyError():
                return Response(status_code=404, content="Model not found")
            case ValueError():
                return Response(status_code=409, content=str(e))
