#model_manager.py
from __future__ import annotations

from collections import OrderedDict
import gc
import json
import logging
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

from cached_path import cached_path
from faster_whisper import WhisperModel

# Assuming these are the model classes and the utility loading function
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import load_model as f5_load_model_utility


if TYPE_CHECKING:
    from collections.abc import Callable


    from .config import (
        WhisperConfig,
    )

logger = logging.getLogger(__name__)
F5CompatibleModel = Union[DiT, UNetT]

F5ModelId = Tuple[str, str, Optional[str], str]

# TODO: enable concurrent model downloads


class SelfDisposingModel[T]:
    def __init__(
        self,
        model_id: str,
        load_fn: Callable[[], T],
        ttl: int,
        model_unloaded_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.load_fn = load_fn
        self.ttl = ttl
        self.model_unloaded_callback = model_unloaded_callback

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.model: T | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.model is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()
            self.model = None
            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.model_unloaded_callback is not None:
                self.model_unloaded_callback(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.model is None
            logger.debug(f"Loading model {self.model_id}")
            start = time.perf_counter()
            self.model = self.load_fn()
            logger.info(f"Model {self.model_id} loaded in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            logger.debug(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.ttl}s")
                    self.expire_timer = threading.Timer(self.ttl, self.unload)
                    self.expire_timer.start()
                elif self.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> T:
        with self.rlock:
            if self.model is None:
                self._load()
            self._increment_ref()
            assert self.model is not None
            return self.model

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class WhisperModelManager:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self.whisper_config = whisper_config
        self.loaded_models: OrderedDict[str, SelfDisposingModel[WhisperModel]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> WhisperModel:
        device = self.whisper_config.inference_device
        compute_type = self.whisper_config.compute_type
        # print(f"Loading Whisper model {model_id} on device {device} with compute type {compute_type}")
        return WhisperModel(
            model_id,
            device=device,
            device_index=self.whisper_config.device_index,
            compute_type=compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
        )
    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992
            self.loaded_models[model_id].unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[WhisperModel]:
        logger.debug(f"Loading model {model_id}")
        with self._lock:
            logger.debug("Acquired lock")
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[WhisperModel](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.whisper_config.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]



class F5TTSModelManager:
    """
    Manages loading, unloading, and caching of F5-TTS and E2-TTS models.
    Uses SelfDisposingModel to handle automatic unloading based on TTL.
    """
    def __init__(self, ttl: int) -> None:
        """
        Initializes the F5TTSModelManager.

        Args:
            ttl: Time-to-live in seconds for idle models.
                 - > 0: Unload after `ttl` seconds of inactivity.
                 - == 0: Unload immediately after use.
                 - < 0: Never unload automatically.
        """
        self.ttl = ttl
        # The key is F5ModelId: (ModelClassName, ckpt_path, vocab_path_or_none, config_json_string)
        self.loaded_models: OrderedDict[F5ModelId, SelfDisposingModel[F5CompatibleModel]] = OrderedDict()
        self._lock = threading.Lock() # Lock for modifying the loaded_models dictionary
        logger.info(f"F5TTSModelManager initialized with TTL={ttl}s")

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Resolves potential 'hf://' paths using cached_path."""
        if path and path.startswith("hf://"):
            try:
                resolved = str(cached_path(path))
                logger.debug(f"Resolved Hugging Face path '{path}' to '{resolved}'")
                return resolved
            except Exception as e:
                logger.error(f"Failed to resolve Hugging Face path '{path}': {e}")
                raise # Re-raise the exception
        elif path and Path(path).exists():
             logger.debug(f"Using local path '{path}'")
             return path
        elif path:
            # Path specified but doesn't exist locally and isn't hf://
             logger.warning(f"Path '{path}' not found locally and doesn't start with hf://. Treating as potentially invalid.")
             # Depending on requirements, you might raise an error here or let f5_load_model_utility handle it.
             # For now, let's pass it through, but log a warning.
             return path # Pass it through, maybe f5_load_model_utility can handle relative paths?
        return None # Path was None initially

    def _get_model_class(self, class_name: str) -> Type[F5CompatibleModel]:
        """Gets the model class object from its name string."""
        if class_name == "DiT":
            return DiT
        elif class_name == "UNetT":
            return UNetT
        else:
            raise ValueError(f"Unknown model class name: {class_name}")

    def _load_fn(self, model_id: F5ModelId) -> F5CompatibleModel:
        """
        The actual loading function called by SelfDisposingModel.
        Uses the details stored in the model_id tuple.
        """
        model_class_name, ckpt_path_orig, vocab_path_orig, config_json = model_id
        logger.info(f"Load function called for model_id: {model_id}")

        model_cls = self._get_model_class(model_class_name)
        ckpt_path = self._resolve_path(ckpt_path_orig)
        vocab_path = self._resolve_path(vocab_path_orig)

        if not ckpt_path or not Path(ckpt_path).exists():
             raise FileNotFoundError(f"Checkpoint file not found after resolving: {ckpt_path} (original: {ckpt_path_orig})")
        if vocab_path and not Path(vocab_path).exists():
            # Only raise if vocab_path was specified but not found after resolving
            raise FileNotFoundError(f"Vocabulary file not found after resolving: {vocab_path} (original: {vocab_path_orig})")

        try:
            model_cfg = json.loads(config_json)
            logger.debug(f"Parsed model config: {model_cfg}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model config JSON: {config_json} - Error: {e}")
            raise ValueError(f"Invalid model config JSON: {e}")

        # Call the utility function from f5_tts.infer.utils_infer
        # Make sure f5_load_model_utility handles device placement etc.
        logger.info(f"Calling f5_load_model_utility with: class={model_cls.__name__}, cfg={model_cfg}, ckpt={ckpt_path}, vocab={vocab_path}")
        try:
            loaded_model = f5_load_model_utility(
                ModelClass=model_cls,
                model_cfg=model_cfg,
                ckpt_path=ckpt_path,
                vocab_file=vocab_path # Pass None if vocab_path is None or empty
            )
            logger.info(f"Model loaded successfully via f5_load_model_utility.")
            return loaded_model
        except Exception as e:
            logger.error(f"Error during f5_load_model_utility execution: {e}", exc_info=True)
            raise # Re-raise the exception

    def _handle_model_unloaded(self, model_id: F5ModelId) -> None:
        """Callback function when a model is successfully unloaded by SelfDisposingModel."""
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"Removing model {model_id} from loaded_models dictionary.")
                del self.loaded_models[model_id]
            else:
                logger.warning(f"Received unload notification for model {model_id}, but it was not found in loaded_models.")

    def unload_model(self, model_id: F5ModelId) -> None:
        """
        Explicitly unloads a model identified by its ID tuple.
        Raises KeyError if the model is not currently managed.
        """
        with self._lock: # Ensure thread safety when accessing loaded_models
            model_wrapper = self.loaded_models.get(model_id)
            if model_wrapper is None:
                raise KeyError(f"Model {model_id} not found in manager.")

            logger.info(f"Explicitly requesting unload for model {model_id}")
            # We ask the wrapper to unload. It will check ref counts internally.
            # If refs > 0, it might log a warning or raise an error depending on its impl.
            try:
                 # Call the wrapper's unload method directly
                 # Note: This bypasses the TTL timer if called explicitly.
                 # It will still respect the ref_count check within unload().
                model_wrapper.unload()
                 # If unload succeeds and calls _handle_model_unloaded,
                 # the entry might already be removed from loaded_models.
                 # Re-checking before deleting defensively.
                if model_id in self.loaded_models:
                     # This case might happen if unload() failed due to ref_count > 0
                     # Or if _handle_model_unloaded wasn't called for some reason.
                     # Check if it's actually loaded before trying to delete again.
                    if not model_wrapper.is_loaded:
                        logger.debug(f"Model {model_id} was already unloaded, removing remnant entry.")
                        del self.loaded_models[model_id]
                    else:
                        logger.warning(f"Explicit unload requested for {model_id}, but it seems still loaded (likely due to active refs).")
                else:
                    logger.info(f"Model {model_id} successfully unloaded and removed from manager.")

            except ValueError as e:
                 # Catch potential error from unload() if ref_count > 0
                 logger.warning(f"Could not explicitly unload model {model_id}: {e}")


    def load_model(
        self,
        model_cls: Type[F5CompatibleModel],
        ckpt_path: str,
        model_cfg: Dict[str, Any],
        vocab_path: Optional[str] = None
    ) -> SelfDisposingModel[F5CompatibleModel]:
        """
        Loads or retrieves a cached F5-TTS/E2-TTS model.

        Args:
            model_cls: The class of the model to load (e.g., DiT, UNetT).
            ckpt_path: Path or hf:// identifier for the model checkpoint.
            model_cfg: Dictionary containing the model configuration.
            vocab_path: Optional path or hf:// identifier for the vocabulary file.

        Returns:
            A SelfDisposingModel wrapper instance. Use 'with' statement to access the model.
        """
        # Ensure config is consistently represented for hashing
        try:
            config_json = json.dumps(model_cfg, sort_keys=True)
        except TypeError as e:
            logger.error(f"Model config dictionary is not JSON serializable: {model_cfg} - Error: {e}")
            raise ValueError(f"Invalid model config: {e}")

        # Normalize paths for the key (use original paths before resolving)
        # Use None for vocab_path in key if it's empty or None
        norm_vocab_path = vocab_path if vocab_path else None
        model_id: F5ModelId = (model_cls.__name__, ckpt_path, norm_vocab_path, config_json)

        logger.debug(f"Request to load model with generated ID: {model_id}")

        with self._lock: # Protect access to loaded_models dict
            if model_id in self.loaded_models:
                logger.debug(f"Model {model_id} found in cache. Returning existing wrapper.")
                # Move to end for LRU behaviour (if desired, though OrderedDict keeps insertion order)
                self.loaded_models.move_to_end(model_id)
                return self.loaded_models[model_id]

            logger.info(f"Model {model_id} not in cache. Creating new SelfDisposingModel wrapper.")
            # Create the wrapper. The actual loading (_load_fn) happens lazily
            # when the 'with' block is entered for the first time.
            new_model_wrapper = SelfDisposingModel[F5CompatibleModel](
                model_id=model_id,
                # Pass a lambda that calls our internal _load_fn with the specific model_id
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            self.loaded_models[model_id] = new_model_wrapper
            return new_model_wrapper

    def get_loaded_model_ids(self) -> list[F5ModelId]:
         """Returns a list of IDs for all models currently managed (may or may not be loaded)."""
         with self._lock:
              return list(self.loaded_models.keys())

    def get_actively_loaded_model_ids(self) -> list[F5ModelId]:
        """Returns a list of IDs for models that are currently loaded into memory."""
        with self._lock:
            return [mid for mid, wrapper in self.loaded_models.items() if wrapper.is_loaded]
