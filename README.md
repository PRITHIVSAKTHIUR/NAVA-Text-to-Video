# **NAVA-Text-to-Video — [hf-spaces](https://huggingface.co/spaces/prithivMLmods/NAVA-Text-to-Video)**

NAVA-Text-to-Video is a sophisticated, experimental audio-visual generation framework powered by **Native Audio-Visual Alignment (NAVA)**. Unlike traditional video generation pipelines that treat audio as a post-processing afterthought, NAVA utilizes a 6.3B parameter architecture to generate synchronized video and high-fidelity audio concurrently within a single unified diffusion process. This application provides a comprehensive web-based interface for end-to-end multimedia creation, supporting complex Text-to-Audio-Video (T2AV) and Image-to-Audio-Video (I2AV) workflows.

The suite is engineered for precision, featuring an integrated **Prompt Rewriter** based on Qwen3-4B that translates simple user inputs into dense, cinematic Chinese captions optimized for the NAVA backbone. With support for multi-speaker reference conditioning (ReDimNet) and high-resolution spatial output (up to 1280x704), NAVA-Text-to-Video represents a frontier in coherent, multi-modal generative AI.

> [!IMPORTANT]
> Model: https://huggingface.co/baidu/NAVA, Code: https://github.com/ernie-research/NAVA

> [!NOTE]
> This demo is experimental. More stable updates will be coming soon.

---

https://github.com/user-attachments/assets/d32b99b5-853b-4366-a80c-481272097c79

<img width="1789" height="1455" alt="image (1)" src="https://github.com/user-attachments/assets/6df7e96a-ace6-437e-acb0-5e59893f0355" />

---

### **Key Features**

* **Native Audio-Visual Synchrony:** Generates video frames and audio waveforms simultaneously, ensuring perfect temporal alignment for actions like speech, environment effects (fire, water), and mechanical movements.
* **Dual-Stage Prompt Engineering:** Includes an autonomous prompt rewriter that converts English or short-form keywords into high-quality, descriptive Chinese prompts. This ensures the best possible activation of the model’s learned cinematic styles and audio-visual cues.
* **Image-to-Video (I2V) Support:** Allows users to provide a reference image to serve as the starting frame, enabling precise control over subject identity and scene composition.
* **Voice Timbre Conditioning:** Supports up to two distinct speaker reference WAV files. The system extracts speaker embeddings via ReDimNet to guide the generated audio’s vocal characteristics.
* **Highly Optimized Architecture:**
* **FlashAttention-2 & 3:** Optimized for high-speed inference on modern GPUs.
* **Sequence Parallel (SP) Inference:** Robust support for multi-GPU distribution via `torchrun` to handle intensive 6.3B parameter denoising.
* **ZeroGPU Compatibility:** Implements fully deferred CUDA initialization and custom patching for `Gemma2` and `transformers` masking to survive restrictive ZeroGPU allocation spikes.


* **Interactive Professional UI:** Wrapped in an "Orange Red" custom theme with a real-time execution log, status indicators, and integrated video playback.

### **Installation and Requirements**

To initialize the NAVA-Text-to-Video environment locally, configure a Python environment equipped with a CUDA-enabled GPU (minimum 24GB VRAM recommended for local runs).

**1. Install Pre-requirements**
Ensure your local system package manager is prepared for modern dependency resolution:

```bash
pip install pip>=26.1.2

```

**2. Install Core Requirements**
Install the deep learning stack, specific FlashAttention wheels, and multimedia processing utilities:

```bash
pip install -r requirements.txt

```

### **Core Requirements List**

The application builds on the following primary dependencies (defined in `requirements.txt`):

```text
torch==2.11.0
torchvision
torchaudio
torchcodec
https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
transformers>=4.56.0
diffusers>=0.35.2
accelerate
vllm>=0.6.0
xfuser>=0.4.0
gradio==6.16.0
spaces
opencv-python
moviepy

```

---

### **Usage**

Launch the application by running the main entry point:

```bash
python app.py --config configs/nava.yaml --ckpt NAVA.safetensors

```

Once the model components are initialized in VRAM, navigate to the local address provided (typically `[http://0.0.0.0:7860/](http://0.0.0.0:7860/)`).

1. **Enter Prompt:** Type a short description of the scene in English or Chinese.
2. **Rewrite (Step 2):** Click **✨ Rewrite Prompt**. This is crucial for English users to get high-quality Chinese dense captions that provide specific audio-visual instructions to the model.
3. **Optional Inputs:**
* Upload an image to the **🖼️ Image Input** accordion to set the scene's first frame.
* Upload reference audio to the **🎤 Speaker Reference** accordion to clone specific voice timbres.


4. **Configure Settings:** Adjust inference steps (default 50) and aspect ratio (16:9, 9:16, or 1:1).
5. **Execute:** Click **🚀 Generate**. The process takes approximately 300 seconds on a standard high-end GPU. The output will be a high-quality `.mp4` file with native synchronized audio.

### **Advanced: Multi-GPU (torchrun) Mode**

For systems with multiple GPUs, you can leverage Sequence Parallelism:

```bash
torchrun --nproc_per_node=N_GPUS app.py --config configs/nava.yaml --ckpt NAVA.safetensors

```

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/NAVA-Text-to-Video/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/NAVA-Text-to-Video.git](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FPRITHIVSAKTHIUR%2FNAVA-Text-to-Video.git)
