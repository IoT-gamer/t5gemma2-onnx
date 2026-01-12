# T5Gemma 2 ONNX Export & Inference

This repository contains a collection of Jupyter Colab Notebooks designed to export the Google [T5Gemma 2](https://blog.google/innovation-and-ai/technology/developers-tools/t5gemma-2/) model (specifically `google/t5gemma-2-270m-270m`) to ONNX format.

It provides pipelines for exporting the text encoder, the multimodal (vision) encoder, and the decoder, along with inference scripts to run the models using `onnxruntime`.

## Features:

* **Decoder Export:** Exports the decoder with a custom wrapper for KV caching to optimize inference.
* **Text Encoder Export:** Exports the standard text-only encoder.
* **Multimodal Encoder Export:** Exports the encoder with support for image inputs (using SigLIP style vision tokens).
* **Custom Patching:** Includes monkey-patches for Hugging Face masking functions to ensure successful tracing and export to ONNX.
* **Inference Examples:** dedicated notebooks for testing text generation and multimodal QA tasks on CPU or GPU

## Notebooks
Each notebook is self-contained and can be run directly in Google Colab.
| Notebook | Description | Link |
| :--- | :---: | ---: |
| `t5gemma2_decoder_onnx_export.ipynb` | Exports the decoder component with KV cache wrappers. | [Link](notebooks/t5gemma2_decoder_onnx_export.ipynb) |
| `t5gemma2_encoder_onnx_export.ipynb` | Export the text-only encoder | [Link](notebooks/t5gemma2_encoder_onnx_export.ipynb) |
| `t5gemma2_multimodal_encoder_onnx_export.ipynb` | Exports the multimodal encoder accepting text and pixel values. | [Link](notebooks/t5gemma2_multimodal_encoder_onnx_export.ipynb) |
| `test_t5gemma2_onnx.ipynb` | Inference script for text-to-text generation | [Link](notebooks/test_t5gemma2_onnx.ipynb) |
| `test_multimodal_t5gemma2_onnx.ipynb` | Inference script for image+text-to-text generation | [Link](notebooks/test_multimodal_t5gemma2_onnx.ipynb) |

## Usage
1. **Exporting the Models**

    You must export the components separately. Run the export notebooks in the following order (depending on your needs):

    1. **Decoder:** Run `t5gemma2_decoder_onnx_export.ipynb` to generate `t5gemma2_decoder.onnx`.
    2. **Encoder (Text):** Run `t5gemma2_encoder_onnx_export.ipynb` to generate `t5gemma2_encoder.onnx`.
    3. **Encoder (Multimodal):** Run `t5gemma2_multimodal_encoder_onnx_export.ipynb` to generate `t5gemma2_encoder_multimodal.onnx`.

    Note: The export scripts utilize `torch.onnx.export` with `opset_version=14` (decoder) or `17` (encoders) and disable `dynamo`.

2. **Running Inference**

    Once exported, you can run inference using the test notebooks. Ensure the paths to your `.onnx` files are correct (default paths assume Google Drive structure).

    **Text Generation:** The `test_t5gemma2_onnx.ipynb` notebook implements a generation loop that handles:
    * Tokenization using `AutoTokenizer`.
    * Encoder execution via ONNX Runtime.
    * Decoder execution with KV Cache management and repetition penalties.

    **Multimodal Generation:** The `test_multimodal_t5gemma2_onnx.ipynb` notebook adds:
    * Construction of multimodal inputs (Image tokens + Text tokens).
    * Execution of the multimodal encoder followed by the standard decoder loop.

## Technical Details
* **Model ID:** `google/t5gemma-2-270m-270m`.
* **Precision:** The multimodal encoder attempts to use `float16` to save RAM during the load process, while the standard text export uses `float32`
* **KV Cache:** The decoder export wraps the model to expose `past_key_values` and `present_key_values`, allowing for efficient stateful generation.
* **Masking:** Custom patches are applied to `transformers.masking_utils.create_bidirectional_mask` to avoid complex tensor logic that fails during ONNX tracing.

## Acknowledgements
* **Model:** T5Gemma-2 by Google DeepMind.
* **Library:** Hugging Face Transformers.