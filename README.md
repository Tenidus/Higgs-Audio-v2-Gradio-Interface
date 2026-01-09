# Higgs-Audio-v2-Gradio-Interface
## This is a Gradio/Web interface for Higgs Audio v2.  Chunking for long-form audio generation included, and multi-speaker.

This is more than your basic web interface for Higgs as it provides the ability to customize all of the options that Higgs v2 has to offer.
This does NOT require a modified installation of Higgs v2, you will simply add a file to the "higgs-audio/examples" directory.  Steps will be listed below.

## Here is a list of the options that are customizable in the GUI:
* model_path - Provides the option to specify the location of custom models.
* audio_tokenizer - Provides the option to specify the location of custom tokenizers.
* max_new_tokens - Slider for setting custom maximum number of new tokens to generate.  Default is 2048
* device - Selectable options for loading on Cuda, MPS, CPU or Auto select.
* use_static_kv_cache - Enable or Disable option
* transcript - Text to be generated into audio.  This supports long-form generations!
* scene_prompt - The scene description prompt to use for generation. (Optional)  Visit: https://github.com/boson-ai/higgs-audio/tree/main/examples for scene prompt details.
* ref_audio - When using for custom voice cloning, ability to specify file path. (Will be adding upload capability).  Currently, place your custom voices to clone (Up to 1 min of audio) in the "higgs-audio/examples/voice_prompts" and use the file name (without file extention) to use custom voice.

### Sampling Parameters:
* temperature - Slider for setting custom temperatures
* top_k - Slider for setting custom Top K
* top_p - Slider for setting custom Top P
* ras_win_len - Slider for setting custom RAS Window Length
* ras_win_max_num_repeat - Slider for setting maximum number of times to repeat the RAS window
* seed - Manually set Seed

### Advanced Options:
* chunk_method - Selectable options for setting text chunking.  Options are "None, Speaker, Word"
* chunk_max_word_num - Slider for setting custom maximum number of words for each chunk.  Only supported when "Word" chunking is selected
* chunk_max_num_turns - Slider for setting custom maximum number of turns for each chunk.  Only supported when "Speaker" chunking is selected
* generation_chunk_buffer_size - Slider for setting custom maximum number of chunks to keep in the buffer.  Reference Audios and 'max_chunk_bugger' chunks are always kept

### Additional Notes:
* Use [SPEAKER0], [SPEAKER1] tags in your transcript
* For sound effects: Use tags like [laugh], [music], [applause]
* Adjust temperature for more/less variation in speech
* Use chunking for very long texts


## **Installation**:
Download the "higgs_audio_gradio.py"
Place it in the "higgs-audio/examples" folder (Replace "higgs-audio" directory for the correct directory name if you have a custom name)

**Activate your python environment:**
### If you used Option 2: Using venv for the install:
```
Navigate to the installation directory
source higgs_audio_env/bin/activate
and run:
pip install gradio
```

### If you used Option 3: Using conda for the install:
```
conda activate ./conda_env
and run:
pip install gradio
```

### If you used Option 4: Using uv for the install:
```
Navigate to the installation directory
source .venv/bin/activate
and run:
uv pip install gradio
```

Running Gradio Interface:
```
Navigate to "higgs-audio/examples" (or custom folder name)
and run:
python higgs_audio_gradio.py
```
Enjoy!
