# whisper_utils.py
from collections.abc import Generator
import logging
import os
from pathlib import Path
import typing
from typing import Optional # Added Optional

import huggingface_hub
from huggingface_hub.utils import HfHubHTTPError # Import specific error

from api_types import Model
import datetime

LIBRARY_NAME = "ctranslate2"
TASK_NAME_TAG = "automatic-speech-recognition"

logger = logging.getLogger(__name__)


def add_custom_whisper_model(
    model_id: str,
    languages: Optional[list[str]] = None,
    created_timestamp: Optional[int] = None,
) -> Model:
    """
    Adds a custom Whisper model by ID. Useful for models that may not be properly 
    tagged on the HF Hub or for local custom models.
    
    Args:
        model_id: The Hugging Face model ID (e.g., 'erax-ai/EraX-WoW-Turbo-V1.1')
        languages: Optional list of language codes supported by the model
        created_timestamp: Optional creation timestamp (defaults to current time)
        
    Returns:
        Model object representing the custom Whisper model
    """
    if created_timestamp is None:
        created_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        
    # Extract owner from the model ID
    owner = model_id.split("/")[0] if "/" in model_id else "local"
    
    return Model(
        id=model_id,
        created=created_timestamp,
        owned_by=owner,
        language=languages or [],
        task=TASK_NAME_TAG,
    )

def _create_model_object_from_hf_info(model_info: huggingface_hub.hf_api.ModelInfo) -> Optional[Model]:
    """Helper to create a Speaches Model object from HuggingFace ModelInfo."""
    # Basic check if it's likely a CTranslate2 Whisper model based on tags/library
    is_whisper = False
    if model_info.library_name == LIBRARY_NAME:
        is_whisper = True
    elif model_info.tags and TASK_NAME_TAG in model_info.tags:
         is_whisper = True
    # Add more checks if needed (e.g., keywords in ID)

    if not is_whisper:
        logger.debug(f"Model {model_info.id} skipped: library_name='{model_info.library_name}', tags='{model_info.tags}'")
        return None

    assert model_info.created_at is not None # Should be present from API
    card_data = model_info.card_data

    language: list[str] = []
    if card_data and card_data.language:
        if isinstance(card_data.language, str):
            language = [card_data.language]
        elif isinstance(card_data.language, list):
             language = card_data.language # Assume list of strings

    return Model(
        id=model_info.id,
        created=int(model_info.created_at.timestamp()),
        owned_by=model_info.id.split("/")[0],
        language=language,
        task=TASK_NAME_TAG,
    )


def list_whisper_models() -> Generator[Model, None, None]:
    """Lists CTranslate2 Whisper models from Hugging Face Hub."""
    # Combine filters for better results
    models_iterator = huggingface_hub.list_models(
        filter=huggingface_hub.ModelFilter(
            task=TASK_NAME_TAG,
            library=LIBRARY_NAME,
        ),
        cardData=True, # Request card data for language info
        sort="downloads",
        direction=-1
    )

    for model_info in models_iterator:
        model_obj = _create_model_object_from_hf_info(model_info)
        
        if model_obj:
            yield model_obj


def list_local_whisper_models() -> Generator[Model, None, None]:
    """Lists locally cached CTranslate2 Whisper models by checking READMEs."""
    try:
        hf_cache = huggingface_hub.scan_cache_dir()
    except huggingface_hub.CacheNotFound:
        logger.warning("Hugging Face cache directory not found. Cannot list local models.")
        return
    except Exception as e:
        logger.error(f"Error scanning Hugging Face cache: {e}", exc_info=True)
        return

    hf_model_repos = [repo for repo in hf_cache.repos if repo.repo_type == "model"]
    logger.debug(f"Found {len(hf_model_repos)} model repositories in local cache.")

    for repo in hf_model_repos:
        try:
            # Find the most likely latest revision available locally
            latest_revision = next(iter(repo.revisions), None)
            if not latest_revision:
                logger.warning(f"No revisions found locally for cached repo: {repo.repo_id}")
                continue

            # Find the README file within that revision's files
            readme_file = next((f for f in latest_revision.files if f.file_name == "README.md"), None)
            readme_path: Optional[Path] = None
            if readme_file and readme_file.file_path.exists():
                readme_path = Path(readme_file.file_path)
            else:
                # Attempt to download if missing (might fail in offline mode)
                logger.debug(f"Local README.md missing for {repo.repo_id}, attempting download.")
                try:
                    readme_path = Path(huggingface_hub.hf_hub_download(
                        repo_id=repo.repo_id,
                        filename="README.md",
                        revision=latest_revision.commit_hash, # Use specific hash
                        repo_type="model",
                        cache_dir=hf_cache.cache_dir, # Ensure it uses the scanned cache
                        local_files_only=os.getenv("HF_HUB_OFFLINE") is not None, # Respect offline flag
                    ))
                except Exception as download_err:
                    logger.warning(f"Could not get README.md for local model {repo.repo_id}: {download_err}")
                    continue # Skip model if README is unavailable

            if not readme_path or not readme_path.exists():
                 logger.warning(f"README.md still not found for {repo.repo_id} after potential download attempt.")
                 continue

            # Load and parse the ModelCard
            model_card = huggingface_hub.ModelCard.load(readme_path)
            # Use .data which should handle missing fields more gracefully
            card_data = model_card.data

            # Check tags and library name from the card data
            is_whisper = False
            if card_data.library_name == LIBRARY_NAME:
                is_whisper = True
            elif card_data.tags and TASK_NAME_TAG in card_data.tags:
                is_whisper = True

            if is_whisper:
                language: list[str] = []
                if card_data.language:
                     if isinstance(card_data.language, str):
                         language = [card_data.language]
                     elif isinstance(card_data.language, list):
                         language = card_data.language

                # Use last_modified timestamp from the repo info as 'created' approximation
                created_timestamp = int(repo.last_modified) if repo.last_modified else 0

                transformed_model = Model(
                    id=repo.repo_id,
                    created=created_timestamp,
                    owned_by=repo.repo_id.split("/")[0],
                    language=language,
                    task=TASK_NAME_TAG,
                )
                yield transformed_model
            else:
                 logger.debug(f"Skipping cached model {repo.repo_id}: Not identified as CTranslate2 Whisper via README.")

        except Exception as e:
            logger.error(f"Error processing cached model {repo.repo_id}: {e}", exc_info=True)
            continue # Skip to next model on error


def get_hf_whisper_model_info(model_id: str) -> Optional[Model]:
    """Fetches info for a specific Whisper model ID from Hugging Face Hub."""
    try:
        logger.debug(f"Querying HF Hub API for model info: {model_id}")
        model_info = huggingface_hub.hf_api.model_info(model_id, files_metadata=False) # Don't need file list
        return _create_model_object_from_hf_info(model_info)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.debug(f"Model ID '{model_id}' not found on Hugging Face Hub.")
            return None
        else:
            logger.error(f"HTTP error fetching model info for '{model_id}' from HF Hub: {e}", exc_info=True)
            return None # Or re-raise depending on desired behavior
    except Exception as e:
        logger.error(f"Unexpected error fetching model info for '{model_id}' from HF Hub: {e}", exc_info=True)
        return None

def get_local_whisper_model_info(model_id: str) -> Optional[Model]:
    """Attempts to get info for a specific locally cached Whisper model ID."""
    # This is less efficient than listing all, but needed for the specific lookup.
    # It iterates through the local models again.
    logger.debug(f"Searching for local model info for ID: {model_id}")
    for local_model in list_local_whisper_models():
        if local_model.id == model_id:
            logger.debug(f"Found local model info for ID: {model_id}")
            return local_model
    logger.debug(f"Local model info not found for ID: {model_id}")
    return None