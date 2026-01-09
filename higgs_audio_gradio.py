"""Gradio interface for HiggsAudio generation."""

import gradio as gr
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import torchaudio
import yaml
import torch

from loguru import logger
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from typing import List, Optional
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict
import tqdm

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def normalize_chinese_punctuation(text):
    """Convert Chinese (full-width) punctuation marks to English (half-width) equivalents."""
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, 
    chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


class HiggsAudioModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device=None,
        device_id=None,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],
        use_static_kv_cache=False,
    ):
        if device_id is not None:
            device = f"cuda:{device_id}"
            self._device = device
        else:
            if device is not None:
                self._device = device
            else:
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        if isinstance(audio_tokenizer, str):
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
        else:
            self._audio_tokenizer = audio_tokenizer

        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self, messages, audio_ids, chunked_text, generation_chunk_buffer_size,
        temperature=1.0, top_k=50, top_p=0.95, ras_win_len=7, 
        ras_win_max_num_repeat=2, seed=123, *args, **kwargs
    ):
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(Message(role="user", content=chunk_text))
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

        if concat_audio_out_ids.device.type == "mps":
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids

        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context for generation."""
    system_message = None
    messages = []
    audio_ids = []
    
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), f"Voice prompt audio file {prompt_audio_path} does not exist."
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(Message(role="assistant", content=AudioContent(audio_url=prompt_audio_path)))
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = []
            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(role="system", content="\n\n".join(system_message_l))
    
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


# Global model client (initialized once)
model_client = None


def initialize_model(model_path, audio_tokenizer_path, max_new_tokens, device, use_static_kv_cache):
    """Initialize the model client."""
    global model_client
    
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=audio_tokenizer_device)
    
    if device == "mps" and use_static_kv_cache:
        use_static_kv_cache = False
    
    model_client = HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer,
        device=device,
        device_id=None,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=use_static_kv_cache,
    )
    return "Model initialized successfully!"


def generate_audio(
    transcript_text,
    scene_prompt_text,
    ref_audio,
    ref_audio_in_system_message,
    max_new_tokens_gen,
    chunk_method,
    chunk_max_word_num,
    chunk_max_num_turns,
    generation_chunk_buffer_size,
    temperature,
    top_k,
    top_p,
    ras_win_len,
    ras_win_max_num_repeat,
    seed,
):
    """Generate audio from text using HiggsAudio."""
    global model_client
    
    if model_client is None:
        return None, "Please initialize the model first!"
    
    # Temporarily update max_new_tokens for this generation
    original_max_new_tokens = model_client._max_new_tokens
    model_client._max_new_tokens = int(max_new_tokens_gen)
    
    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    transcript = transcript_text.strip()
    scene_prompt = scene_prompt_text.strip() if scene_prompt_text else None
    
    speaker_tags = sorted(set(pattern.findall(transcript)))
    transcript = normalize_chinese_punctuation(transcript)
    transcript = transcript.replace("(", " ").replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)
    
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=ref_audio if ref_audio else None,
        ref_audio_in_system_message=ref_audio_in_system_message,
        audio_tokenizer=model_client._audio_tokenizer,
        speaker_tags=speaker_tags,
    )
    
    chunked_text = prepare_chunk_text(
        transcript,
        chunk_method=chunk_method if chunk_method != "None" else None,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
    )

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=generation_chunk_buffer_size if generation_chunk_buffer_size > 0 else None,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=seed if seed > 0 else None,
    )

    output_path = "gradio_output.wav"
    sf.write(output_path, concat_wv, sr)
    
    # Restore original max_new_tokens
    model_client._max_new_tokens = original_max_new_tokens
    
    return output_path, f"Audio generated successfully!\nSample rate: {sr} Hz"


# Create Gradio interface
with gr.Blocks(title="HiggsAudio Generator") as demo:
    gr.Markdown("# HiggsAudio Text-to-Speech Generator")
    gr.Markdown("Generate high-quality speech from text using HiggsAudio models.")
    
    with gr.Tab("Model Setup"):
        model_path = gr.Textbox(
            label="Model Path",
            value="bosonai/higgs-audio-v2-generation-3B-base",
            info="Hugging Face model path or local directory"
        )
        audio_tokenizer_path = gr.Textbox(
            label="Audio Tokenizer Path",
            value="bosonai/higgs-audio-v2-tokenizer",
            info="Audio tokenizer model path"
        )
        max_new_tokens = gr.Slider(
            minimum=256, maximum=8192, value=2048, step=256,
            label="Max New Tokens",
            info="Maximum number of tokens to generate"
        )
        device = gr.Radio(
            choices=["auto", "cuda", "mps", "cpu"],
            value="auto",
            label="Device",
            info="Computation device (auto will pick the best available)"
        )
        use_static_kv_cache = gr.Checkbox(
            label="Use Static KV Cache",
            value=True,
            info="Enable static KV cache for faster generation (GPU only)"
        )
        init_btn = gr.Button("Initialize Model", variant="primary")
        init_status = gr.Textbox(label="Status", interactive=False)
        
        init_btn.click(
            fn=initialize_model,
            inputs=[model_path, audio_tokenizer_path, max_new_tokens, device, use_static_kv_cache],
            outputs=init_status
        )
    
    with gr.Tab("Generate Audio"):
        with gr.Row():
            with gr.Column():
                transcript_text = gr.Textbox(
                    label="Transcript",
                    placeholder="Enter the text to convert to speech...",
                    lines=10,
                    info="Use [SPEAKER0], [SPEAKER1] tags for multi-speaker"
                )
                scene_prompt_text = gr.Textbox(
                    label="Scene Prompt (Optional)",
                    placeholder="Describe the acoustic environment...",
                    lines=3
                )
                ref_audio = gr.Textbox(
                    label="Reference Audio (Optional)",
                    placeholder="e.g., belinda or belinda,chadwick for multi-speaker",
                    info="Leave empty to let model choose voice"
                )
                ref_audio_in_system = gr.Checkbox(
                    label="Include Reference Audio in System Message",
                    value=False
                )
                
            with gr.Column():
                gr.Markdown("### Sampling Parameters")
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                    label="Temperature",
                    info="Higher values = more random"
                )
                top_k = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Top K",
                    info="Number of top tokens to consider"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top P",
                    info="Cumulative probability threshold"
                )
                ras_win_len = gr.Slider(
                    minimum=0, maximum=20, value=7, step=1,
                    label="RAS Window Length",
                    info="0 to disable RAS sampling"
                )
                ras_win_max_num_repeat = gr.Slider(
                    minimum=1, maximum=10, value=2, step=1,
                    label="RAS Max Repeats"
                )
                seed = gr.Number(
                    label="Seed",
                    value=123,
                    precision=0,
                    info="Set to 0 for random seed"
                )
        
        with gr.Accordion("Advanced Options", open=False):
            max_new_tokens_gen = gr.Number(
                label="Max New Tokens",
                value=2048,
                precision=0,
                info="Maximum number of new tokens to generate for this audio"
            )
            chunk_method = gr.Radio(
                choices=["None", "speaker", "word"],
                value="None",
                label="Chunk Method",
                info="How to split long text"
            )
            chunk_max_word_num = gr.Slider(
                minimum=50, maximum=500, value=200, step=50,
                label="Max Words per Chunk",
                info="Used when chunk method is 'word'"
            )
            chunk_max_num_turns = gr.Slider(
                minimum=1, maximum=10, value=1, step=1,
                label="Max Turns per Chunk",
                info="Used when chunk method is 'speaker'"
            )
            generation_chunk_buffer_size = gr.Slider(
                minimum=0, maximum=20, value=0, step=1,
                label="Generation Chunk Buffer Size",
                info="0 to keep all chunks in context"
            )
        
        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")
        
        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            status_output = gr.Textbox(label="Generation Status", interactive=False)
        
        generate_btn.click(
            fn=generate_audio,
            inputs=[
                transcript_text, scene_prompt_text, ref_audio, ref_audio_in_system,
                max_new_tokens_gen, chunk_method, chunk_max_word_num, chunk_max_num_turns,
                generation_chunk_buffer_size, temperature, top_k, top_p,
                ras_win_len, ras_win_max_num_repeat, seed
            ],
            outputs=[audio_output, status_output]
        )
    
    gr.Markdown("""
    ### Usage Tips:
    1. **Initialize the model first** in the "Model Setup" tab
    2. For multi-speaker: Use `[SPEAKER0]`, `[SPEAKER1]` tags in your transcript
    3. For sound effects: Use tags like `[laugh]`, `[music]`, `[applause]`
    4. Adjust temperature for more/less variation in speech
    5. Use chunking for very long texts
    """)


if __name__ == "__main__":
    demo.launch(share=True)