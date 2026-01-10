# Higgs-Audio-v2-Gradio-Interface
## This is a Gradio/Web interface for Higgs Audio v2.  Chunking for long-form audio generation included, and multi-speaker.

## Generate Audio tab Options - New Ability to Upload Text to Speech from .TXT file
* Will automatically read and clean the text (removes indents, normalizes spacing, removes improper characters)
* Removes leading/trailing whitespace and tabs from each line
* Preserves SPEAKER tags and other formatting needed for TTS
<img width="1850" height="781" alt="Screenshot_4" src="https://github.com/user-attachments/assets/3fbbdce2-3f67-4d10-b4f8-cd4996480f72" />




This is more than your basic web interface for Higgs as it provides the ability to customize all of the options that Higgs v2 has to offer.  
This does NOT require a modified installation of Higgs v2, you will simply add a file to the "higgs-audio/examples" directory.  Steps will be listed below.  
Before Text to Speech generation, it will automatically load and unload the model from memory.


## Here is a list of the options that are customizable in the GUI:
* model_path - Provides the option to specify the location of custom models.
* audio_tokenizer - Provides the option to specify the location of custom tokenizers.
* max_new_tokens - Slider for setting custom maximum number of new tokens to generate.  Default is 2048
* device - Selectable options for loading on Cuda, MPS, CPU or Auto select.
* use_static_kv_cache - Enable or Disable option
* transcript - Text to be generated into audio.  This supports long-form generations!
* scene_prompt - The scene description prompt to use for generation. (Optional)  Visit: https://github.com/boson-ai/higgs-audio/tree/main/examples for scene prompt details.
* ref_audio - When using for custom voice cloning.  Ability to select from standard/built-in voices or upload your own custom voice.  Uploading a custom voice will place it in the default `higgs-audio/examples/voice_prompts` folder.

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
* Model auto-loads before generation and auto-unloads after to save memory
* You can manually initialize the model in "Model Setup" to keep it loaded between generations
* Select a voice prompt from the dropdown or choose "None" for random voice
* Upload a .txt file to automatically populate the transcript (you can still edit it)
* For multi-speaker: Use [SPEAKER0], [SPEAKER1] tags in your transcript
* For sound effects: Use tags like [laugh], [music], [applause]
* Adjust temperature for more/less variation in speech
* Use chunking for very long texts


## **Installation**:
Download the "higgs_audio_gradio.py"
Place it in the "higgs-audio/examples" folder (Replace "higgs-audio" directory for the correct directory name if you have a custom location)

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

# Screenshots:
## Model Setup tab
<img width="1488" height="781" alt="Screenshot_1" src="https://github.com/user-attachments/assets/4467a513-625c-48cd-b92f-9c541c0965dc" />

## Generate Audio tab Options - New Ability to Upload Custom Voices
<img width="925" height="908" alt="Screenshot_5" src="https://github.com/user-attachments/assets/ec1e8c1d-fdbe-4ea5-bbca-d8bc3640526f" />

## Advanced Options (Under Generate Audio tab)
<img width="1484" height="876" alt="Screenshot_6" src="https://github.com/user-attachments/assets/38b37eb9-b37f-47d3-a4d6-c5fdbda8884e" />
