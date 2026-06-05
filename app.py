import sys
import types
from importlib.machinery import ModuleSpec

from typing import Iterable

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

class DummyLayerRepository:
    def __init__(self, *args, **kwargs):
        pass

kernels = types.ModuleType("kernels")
kernels_layer = types.ModuleType("kernels.layer")
kernels_layer_layer = types.ModuleType("kernels.layer.layer")

# Set __spec__ to prevent ValueError: kernels.__spec__ is None in python 3.12
kernels.__spec__ = ModuleSpec("kernels", None, is_package=True)
kernels_layer.__spec__ = ModuleSpec("kernels.layer", None, is_package=True)
kernels_layer_layer.__spec__ = ModuleSpec("kernels.layer.layer", None, is_package=False)
kernels.__version__ = "0.0.1"

kernels_layer_layer.LayerRepository = DummyLayerRepository
kernels_layer.LayerRepository = DummyLayerRepository
kernels_layer.layer = kernels_layer_layer
kernels.layer = kernels_layer

sys.modules["kernels"] = kernels
sys.modules["kernels.layer"] = kernels_layer
sys.modules["kernels.layer.layer"] = kernels_layer_layer

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import math
import time
import argparse
import datetime
import importlib

import torch
import yaml

_IS_ZERO_GPU = (
    os.environ.get("SPACES_ZERO_GPU", "0") == "1"
    or "spaces" in sys.modules
    or os.path.exists("/usr/local/lib/python3.12/site-packages/spaces")
)

try:
    import spaces as _spaces_module
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False
    _spaces_module = None

def _spaces_gpu_decorator(fn=None, *, size="xlarge", duration=120):
    """
    Conditional @spaces.GPU decorator.
    - On ZeroGPU: applies the real spaces.GPU decorator.
    - Elsewhere:  returns the function unchanged (no-op).
    """
    def _decorator(f):
        if _HAS_SPACES and _IS_ZERO_GPU:
            return _spaces_module.GPU(size=size, duration=duration)(f)
        return f
    if fn is not None:
        return _decorator(fn)
    return _decorator


ASPECT_RATIO_MAP = {
    "16:9 (1280×704)": (704, 1280),
    "9:16 (704×1280)": (1280, 704),
    "1:1 (960×960)": (960, 960),
}

CMD_INFER = 1
CMD_EXIT = 0


def broadcast_string(s: str, src: int = 0):
    """Broadcast a string from src rank to all ranks."""
    import torch.distributed as dist
    if dist.get_rank() == src:
        data = s.encode("utf-8")
        length = torch.tensor([len(data)], dtype=torch.long, device="cuda")
    else:
        length = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(length, src=src)
    n = length.item()

    if n == 0:
        return ""

    if dist.get_rank() == src:
        tensor = torch.tensor(list(data), dtype=torch.uint8, device="cuda")
    else:
        tensor = torch.empty(n, dtype=torch.uint8, device="cuda")

    dist.broadcast(tensor, src=src)

    if dist.get_rank() != src:
        s = bytes(tensor.cpu().tolist()).decode("utf-8")
    return s


def broadcast_cmd(cmd: int, src: int = 0):
    """Broadcast a command integer from src to all ranks."""
    import torch.distributed as dist
    t = torch.tensor([cmd], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()


def broadcast_int(val: int, src: int = 0):
    """Broadcast a single integer."""
    import torch.distributed as dist
    t = torch.tensor([val], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return t.item()


SYSTEM_PROMPT = """你是一个中文音视频生成 prompt rewriter。你的任务是把用户输入的简短描述、关键词或普通 prompt，改写成一个适合音视频生成模型使用的高质量中文长 prompt。最终只输出改写后的 prompt，不要解释，不要分析，不要输出标题，不要输出 JSON，不要换行，必须是单段中文文本。

你必须保留用户输入中的核心意图，包括主体、动作、速度、情绪、场景、台词和镜头要求。不能把用户指定的动作改成相反含义，不能删除关键主体，不能新增与用户意图冲突的剧情。用户没有明确说明的信息，可以根据画面和常识合理补全，例如背景、光线、镜头、动作细节、环境反馈和音效。

改写后的 prompt 必须具有电影化、具体、连续、可执行的风格。整体结构按以下顺序自然组织：第一，描述视频风格、核心氛围和主体所在场景；第二，描述主体的外观、服装、材质、表情、姿态、位置和整体气质；第三，描述背景环境、远景元素、光线、色调和整体氛围；第四，描述动作过程，必须使用清晰的时间线，包含"视频开始时……随后……随着动作持续……视频结束时……"这类表达；第五，描述镜头语言，包括景别、机位、角度、镜头运动、稳定性、是否切镜，以及镜头重点捕捉的细节；第六，描述对白或无对白；第七，描述音频设计，包括主体动作声、环境声、细节声、空间混响和整体听感。

开头优先使用类似句式："这是一段充满【风格/情绪】与【核心氛围】的视频，画面中【主体】正位于【场景】中……"。如果是写实人物或日常场景，可以使用"这段写实电影风格的视频记录了一个……场景……"。如果是动漫人物，可以使用"画面呈现高质量动漫电影质感……"。如果是运动场景，可以突出阳光、速度感、运动张力和真实临场感。如果是机甲、巨龙、怪兽、赛博人物等场景，可以突出史诗感、压迫感、力量感、未来感或毁灭感。

只要用户提供了台词，必须保留台词内容，不能做任何翻译，必须保留英文原文，必须用 <S> 和 <E> 包裹每句台词，用户给的所有连贯的speech只需要一对<S><E>，不允许在其中插入新的。有多个说话人时，要说明谁先说、谁回应、各自的位置、音色、情绪和声场；如果某个角色不说话，也要明确"全程不说话"。对话类音频要强调清晰近场人声、口型同步、环境底噪、声场定位和混音干净。

如果用户没有明确提供台词，必须写："画面中没有人物对白，也没有任何旁白。" 然后进入纯音效设计。音频设计必须具体，不能只写"有声音"或"有环境音"。纯音效场景要写清楚主体动作声、接触摩擦声、环境声、细节声和空间回响。例如海浪翻卷声、冲浪板切水声、风切声、水花拍打声、发动机轰鸣声、轮胎摩擦声、液压装置声、金属关节摩擦声、火焰喷射声、冰晶碰撞声、低频咆哮声、脚步声、衣料摩擦声、室内混响等。默认不要加入明显背景音乐，除非用户明确要求。结尾必须用类似句式总结："整体听感【听感关键词】，突出【核心体验】。" 或 "整体氛围【氛围关键词】，营造出【目标效果】。"

动作描写必须是视频过程，而不是静态描述。要写清楚主体从什么状态开始，接着如何运动，动作速度如何，动作对环境产生什么影响，最后停留在什么状态。例如，快速动作要体现"迅速、猛烈、强烈、连续、背景快速后掠、浪花炸开、灰尘扬起、装甲联动加快"等细节；慢速动作要体现"缓慢、平稳、克制、柔和、细微调整、节奏舒展、环境变化轻柔"等细节。动作和环境反馈要匹配，例如冲浪要有水花和浪声，机甲要有金属关节和脚步震动，巨龙喷火要有火焰、热浪和火星，吐冰要有冰雾、冰晶和寒风，人物说话要有口型同步和近场人声。

镜头语言要具体。默认使用稳定镜头，不要频繁切镜。根据动作选择合理镜头：高速运动使用低角度侧前方跟拍或稳定跟随；慢速运动使用平稳跟拍并保持固定距离；正面凝视使用中景到中近景、轻微仰视或平视、稳定凝视和轻微推进；喷火、吐冰、大吼使用正面中近景、低角度、锁定嘴部和面部；双人对话使用固定中近景，两人同时入画；日常说话使用近景或中近景，强调口型同步和表情。镜头段落中要使用类似句式："镜头采用稳定的【景别/角度】构图……全程……不切镜、不摇移……细腻捕捉……突出……"。

输出要求：只输出最终改写后的 prompt；必须保留原始speech部分不能忽略 !；必须是中文；必须保留原始speech部分不能忽略；必须是单段；不要换行；不要列表；不要解释；不要加标题；不要输出 JSON；不要使用 markdown；不要出现"根据用户输入""改写如下"等说明性文字。

思考要求：你只需要进行一轮简短思考（分析用户意图、确定风格和结构），然后立即输出最终 prompt。禁止反复推敲、多轮修改或自我检查。思考结束后直接给出最终结果，不要再回头修改。"""


def _to01(x):
    """Convert [-1, 1] tensor to [0, 1]."""
    return torch.clamp((x.float() + 1.0) / 2.0, 0.0, 1.0)


def _toWav(x):
    """Normalize waveform to [-0.95, 0.95] range."""
    peak = x.abs().max().clamp(min=1e-12)
    x = x * (0.95 / peak)
    return x.clamp(-1.0, 1.0)


def _count_speech_tags(text: str) -> int:
    """Count number of <S>...<E> pairs in text."""
    return len(re.findall(r"<S>.*?<E>", text, re.DOTALL))


def ensure_weights():
    """Ensure all required large checkpoints are downloaded to the workspace on CPU before launching Gradio."""
    repo_id = "ernie-research/NAVA"
    
    files = [
        "NAVA.safetensors",
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model",
        "Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json",
        "params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors",
    ]
    
    print("=" * 60)
    print("  NAVA — Checking and downloading model weights (CPU/Startup)...")
    print("=" * 60)
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Warning] huggingface_hub library is not installed. Skipping automatic download.")
        return

    import shutil

    for f in files:
        # Check if local file exists
        if os.path.exists(f):
            print(f"[Weights] Found: {f}")
            continue
            
        # Special check for audio vae path mismatch
        if f == "params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors" and os.path.exists("huggingface_upload/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"):
            print("[Weights] Found audio VAE in huggingface_upload/params/")
            continue

        print(f"[Weights] Downloading {f} from Hugging Face ({repo_id})...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=f,
                local_dir=".",
                local_dir_use_symlinks=False,
            )
            print(f"[Weights] Successfully downloaded {f}")
        except Exception as e:
            print(f"[Weights] Error downloading {f}: {e}")

    # Synchronize audio VAE paths
    src_audio_vae = "params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"
    dst_audio_vae = "huggingface_upload/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors"
    
    if os.path.exists(src_audio_vae) and not os.path.exists(dst_audio_vae):
        print(f"[Weights] Aligning audio VAE: copying {src_audio_vae} to {dst_audio_vae}...")
        os.makedirs(os.path.dirname(dst_audio_vae), exist_ok=True)
        try:
            shutil.copy2(src_audio_vae, dst_audio_vae)
        except Exception as e:
            print(f"[Weights] Copy error: {e}")
            try:
                os.symlink(os.path.abspath(src_audio_vae), os.path.abspath(dst_audio_vae))
            except Exception as sym_err:
                print(f"[Weights] Symlink error: {sym_err}")

    if os.path.exists(dst_audio_vae) and not os.path.exists(src_audio_vae):
        print(f"[Weights] Aligning audio VAE: copying {dst_audio_vae} to {src_audio_vae}...")
        os.makedirs(os.path.dirname(src_audio_vae), exist_ok=True)
        try:
            shutil.copy2(dst_audio_vae, src_audio_vae)
        except Exception as e:
            print(f"[Weights] Copy error: {e}")

    print("=" * 60)
    print("  NAVA — Weights check completed.")
    print("=" * 60)


class NAVAEngine:
    """
    NAVA inference engine wrapper.
    Handles pipeline init, checkpoint loading, SP patching, and single-sample generation.
    Supports: text-to-AV, image-to-AV (i2v), up to 2 speaker reference WAVs.

    IMPORTANT: All CUDA operations are deferred to generate() so that this class
    can be instantiated safely before ZeroGPU assigns a GPU.
    """

    def __init__(self, config_path: str, ckpt_path: str,
                 rank: int = 0, world_size: int = 1, use_sp: bool = False,
                 height: int = 704, width: int = 1280, frames: int = 37):
        """
        Store all init params. Actual model loading is deferred to _lazy_init()
        which is called inside generate() (inside @spaces.GPU scope on ZeroGPU).
        """
        self.config_path = config_path or "configs/nava.yaml"
        self.ckpt_path = ckpt_path or "NAVA.safetensors"
        self.rank = rank
        self.world_size = world_size
        self.use_sp = use_sp
        self.height = height
        self.width = width
        self.frames = frames

        # Will be populated by _lazy_init()
        self.pipe = None
        self.cfg = None
        self.device = None
        self.dtype = None
        self._initialized = False
        self._backbone_on_gpu = False

        print(
            f"[Engine] NAVAEngine created (deferred init). "
            f"rank={rank}, world_size={world_size}, use_sp={use_sp}, "
            f"resolution={width}x{height}, frames={frames}"
        )

    def _lazy_init(self):
        """
        Actually load the model. Called on the first generate() call,
        which is guaranteed to be inside a @spaces.GPU scope on ZeroGPU.
        On torchrun (multi-GPU), dist is already initialized before this call.
        """
        if self._initialized:
            return

        import torchaudio  # noqa: F401 — ensure torchaudio is importable
        from nava_src.utils.common import set_seed
        from nava_src.models.nava.utils.model_loading_utils import load_fusion_checkpoint

        # Resolve device — on ZeroGPU, cuda:0 is the assigned GPU.
        # On torchrun, LOCAL_RANK tells us the device.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        # Load config
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: '{self.config_path}'. Please specify a valid config path via --config."
            )
        self.cfg = yaml.safe_load(open(self.config_path, "r"))
        self.modality = self.cfg.get("modality", "audio_video")
        self.dtype = torch.bfloat16 if self.cfg["use_bf16"] else torch.float16

        set_seed(self.cfg.get("seed", 42))

        # SP init (multi-GPU torchrun only)
        if self.use_sp:
            import torch.distributed as dist
            from nava_src.models.nava.distributed_comms.parallel_states import (
                initialize_sequence_parallel_state,
            )
            initialize_sequence_parallel_state(self.world_size)
            if self.rank == 0:
                print(f"[SP] Sequence parallel enabled, sp_size={self.world_size}")

        # Load pipeline class
        module_path, class_name = self.cfg["pipeline"].rsplit(".", 1)
        PipelineClass = getattr(importlib.import_module(module_path), class_name)
        if "video" in self.modality and "audio" in self.modality:
            self.cfg["init_from_meta"] = True

        self.pipe = PipelineClass.create(
            model_id=self.cfg["model_id"],
            use_bf16=self.cfg["use_bf16"],
            audio_latent_ch=self.cfg["audio_latent_ch"],
            video_latent_ch=self.cfg["video_latent_ch"],
            lambda_ddpm=self.cfg["lambda_ddpm"],
            cfg=self.cfg,
            device=self.device,
        )

        # Resolve checkpoint path — prefer .safetensors, fall back to .ckpt
        ckpt_path = self.ckpt_path
        if not os.path.exists(ckpt_path):
            ckpt_fallback = os.path.splitext(ckpt_path)[0] + ".ckpt"
            if os.path.exists(ckpt_fallback):
                if self.rank == 0:
                    print(f"[Engine] {ckpt_path} not found, falling back to {ckpt_fallback}")
                ckpt_path = ckpt_fallback
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt_path} (also tried {ckpt_fallback}). "
                    f"Please verify the checkpoint exists or specify a valid checkpoint via --ckpt."
                )

        # Load checkpoint weights
        if ("video" in self.modality and "audio" in self.modality
                and not self.cfg.get("use_mmdit_model", False)):
            load_fusion_checkpoint(
                self.pipe.model, checkpoint_path=ckpt_path, from_meta=True
            )
        else:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file as _sf_load
                state_dict = _sf_load(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = self.pipe.model.load_state_dict(state_dict, strict=False)
            if self.rank == 0:
                print(f"[Engine] missing: {missing}, unexpected: {unexpected}")

        self.pipe = self.pipe.to(self.device)
        self.pipe.model.eval()
        self.pipe.model.backbone.set_rope_params()

        # SP patching
        if self.use_sp:
            self._convert_backbone_to_sp()
            if self.rank == 0:
                print("[SP] Patched backbone blocks to SP-aware self-attn.")

        # Infer misc params
        self.fps = self.cfg["data"].get("video_fps", 24)
        self.audio_tokens_per_sec = self.cfg["data"].get("audio_tokens_per_sec", 25)
        self.video_latent_ch = self.cfg["video_latent_ch"]
        self.patch_size = self.cfg.get("spatial_downsample", 16)
        self.resolution = (
            self.pipe.video_vae.resolution
            if hasattr(self.pipe.video_vae, "resolution")
            else 960
        )

        # Offload backbone to CPU immediately after init
        self.pipe.model.backbone.to("cpu")
        torch.cuda.empty_cache()
        self._backbone_on_gpu = False

        self._initialized = True

        if self.rank == 0:
            print(
                f"[Engine] Initialized. modality={self.modality}, "
                f"resolution={self.width}x{self.height}, frames={self.frames}"
            )
            print("[Engine] Backbone offloaded to CPU (will reload to GPU on generate)")

    def _convert_backbone_to_sp(self):
        """In-place swap every block.self_attn to its SP-aware subclass."""
        from nava_src.models.nava.modules.model_mm_sp import (
            WanDoubleStreamSelfAttentionSP,
            WanSelfAttentionSP,
            _swap_self_attn,
        )
        backbone = self.pipe.model.backbone
        for blk in list(backbone.double_blocks) + list(backbone.double_final_blocks):
            _swap_self_attn(blk, WanDoubleStreamSelfAttentionSP)
        for blk in backbone.single_blocks:
            _swap_self_attn(blk, WanSelfAttentionSP)

    def reload_backbone(self):
        """Move backbone to GPU for diffusion sampling."""
        if not self._backbone_on_gpu:
            self.pipe.model.backbone.to(self.device)
            self._backbone_on_gpu = True

    def offload_backbone(self):
        """Move backbone to CPU to free GPU memory."""
        if self._backbone_on_gpu:
            self.pipe.model.backbone.to("cpu")
            torch.cuda.empty_cache()
            self._backbone_on_gpu = False

    def _get_spk_embs(self, spk_wav_paths: list) -> list:
        """
        Get speaker embeddings from local WAV files via ReDimNet speaker model.
        Returns list of Tensor(1, 192), same format as T2AVDataset.
        """
        spk_embs_list = []
        for wav_path in spk_wav_paths:
            if not wav_path or not os.path.exists(wav_path):
                spk_embs_list.append(torch.zeros((1, 192), dtype=torch.float32))
                continue

            query = {
                "data_path": wav_path,
                "use_spk_emb": True,
            }
            result = self.pipe.audio_vae.encode(query).latent_dist.sample()
            spk_embs = result["spk_embs"]  # Tensor(1, 192)
            spk_embs_list.append(spk_embs)

        return spk_embs_list

    def _get_first_frame(self, image_path: str, target_height: int = None,
                         target_width: int = None):
        """
        Encode first frame image via local video VAE.
        Returns img_latents tensor [1, h_latent, w_latent, z_dim].
        """
        img_latents = self.pipe.video_vae.encode(
            image_path, target_height=target_height, target_width=target_width
        ).latent_dist.sample()
        return img_latents

    def _build_batch(self, prompt: str, image_path: str = None,
                     spk_wav_paths: list = None, is_i2v: bool = False,
                     height: int = None, width: int = None):
        """Build a single-sample batch dict from raw inputs."""
        height = height or self.height
        width = width or self.width
        h = height // self.patch_size
        w = width // self.patch_size
        frames = self.frames

        # Audio length based on video duration
        video_duration = ((frames - 1) * 4 + 1) / self.fps
        audio_len = math.ceil(video_duration * self.audio_tokens_per_sec)

        # Default video latents (random noise, shape determines output size)
        video_latents = torch.randn((frames, h, w, 48))

        # Handle first frame (i2v)
        img_latents = None
        if is_i2v and image_path and os.path.exists(image_path):
            img_latents = self._get_first_frame(
                image_path, target_height=height, target_width=width
            )
            video_latents = torch.randn(
                (frames, img_latents.shape[1], img_latents.shape[2], 48)
            )

        audio_latents = torch.randn((audio_len, 48))

        # Handle speaker embeddings (0-2 speakers)
        spk_embs = None
        if spk_wav_paths:
            valid_paths = [p for p in spk_wav_paths if p and os.path.exists(p)]
            if valid_paths:
                spk_embs = self._get_spk_embs(valid_paths)

        # Insert <extra_id_2> after <S> for spk_pos detection (align with T2AVDataset)
        prompt = prompt.replace("<S>", "<S><extra_id_2>")

        batch = {
            "idx": 0,
            "video_latents": video_latents,
            "first_frames": img_latents,
            "audio_latents": audio_latents,
            "save_path": "gradio_output.mp4",
            "captions": prompt,
            "spk_embs": spk_embs,
        }

        return batch

    def _collate_single(self, sample: dict) -> dict:
        """Collate a single sample into batch format (mimics collate_fn for bs=1)."""
        from nava_src.data.t2v import collate_fn
        return collate_fn([sample])

    @torch.no_grad()
    def generate(self, prompt: str, image_path: str = None, spk_wav_paths: list = None,
                 steps: int = 50, output_dir: str = "/tmp/nava_outputs",
                 is_i2v: bool = False, height: int = None, width: int = None,
                 frames: int = None) -> str:
        """
        Run single inference.
        All ranks must call this together in SP mode.
        On ZeroGPU, this must be called inside a @spaces.GPU decorated function.
        Returns: output video path (only meaningful on rank 0).
        """
        # Lazy init — safe here because we are inside @spaces.GPU scope
        self._lazy_init()

        from nava_src.utils.common import set_seed

        # Pick a fresh random seed. In SP mode all ranks must use the SAME seed.
        if self.use_sp:
            import torch.distributed as dist
            seed_t = torch.empty(1, dtype=torch.long, device=self.device)
            if self.rank == 0:
                seed_t.fill_(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            dist.broadcast(seed_t, src=0)
            seed = int(seed_t.item())
        else:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        if self.rank == 0:
            print(f"[Engine] Random seed for this request: {seed}")
        set_seed(seed)

        # Sync all ranks before inference
        if self.use_sp:
            import torch.distributed as dist
            torch.cuda.empty_cache()
            dist.barrier()

        # Per-request frames override
        orig_frames = self.frames
        if frames is not None:
            self.frames = frames

        os.makedirs(output_dir, exist_ok=True)

        sample = self._build_batch(prompt, image_path, spk_wav_paths, is_i2v,
                                   height=height, width=width)
        batch = self._collate_single(sample)
        batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        amp_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)

        # Reload backbone to GPU for diffusion sampling
        self.reload_backbone()

        with amp_ctx:
            gen_vid_out, gen_aud_out = self.pipe.sample(
                batch,
                num_steps=steps,
                audio_guidance_scale=self.cfg.get("audio_guidance_scale", 2.0),
                video_guidance_scale=self.cfg.get("video_guidance_scale", 3.0),
                align_3d_cfg=self.cfg.get("align_3d_cfg", True),
                audio_align_guidance_scale=self.cfg.get("audio_align_guidance_scale", 2.0),
                video_align_guidance_scale=self.cfg.get("video_align_guidance_scale", 3.0),
                save_vid_latent=False,
                is_i2v=is_i2v,
                timbre_cfg=self.cfg.get("timbre_cfg", False),
                timbre_align_guidance_scale=self.cfg.get(
                    "timbre_align_guidance_scale", 3.0
                ),
                offload_backbone=True,
                vae_cpu_offload=False,
                decode=(self.rank == 0),
            )

        self._backbone_on_gpu = True

        # Barrier so workers don't race ahead
        if self.use_sp:
            import torch.distributed as dist
            dist.barrier()

        # Restore original frames setting
        self.frames = orig_frames

        # Only rank 0 saves
        if self.rank != 0:
            return ""

        # Post-process: merge video + audio → mp4
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(output_dir, f"output_{timestamp}.mp4")

        gen_vids = _to01(gen_vid_out).float()
        video_tensor = (gen_vids[0] * 255).clamp(0, 255).to(torch.uint8)
        video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        aud = gen_aud_out[0]
        waveform = _toWav(aud["waveform"])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = aud["sample_rate"]

        # Write video+audio via ffmpeg (torchvision.io.write_video is
        # unavailable on some HF Spaces torchvision builds).
        import subprocess
        import wave
        import numpy as np

        T, H, W, C = video_tensor.shape
        wav_tmp = os.path.join(output_dir, f"_tmp_audio_{timestamp}.wav")

        # Write WAV using stdlib (avoids torchaudio/torchcodec dependency)
        wav_data = waveform.cpu().float().contiguous()
        if wav_data.dim() == 1:
            wav_data = wav_data.unsqueeze(0)
        n_channels = wav_data.shape[0]
        # Convert float32 [-1, 1] to int16
        wav_np = (wav_data.numpy() * 32767).clip(-32768, 32767).astype(np.int16)
        # Interleave channels: [channels, samples] -> [samples, channels] -> flat
        wav_np = wav_np.T  # [samples, channels]
        with wave.open(wav_tmp, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(wav_np.tobytes())

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            # video: raw frames via pipe
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "pipe:0",
            # audio
            "-i", wav_tmp,
            # encoding
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            raw_frames = video_tensor.cpu().numpy().tobytes()
            _, stderr = proc.communicate(input=raw_frames, timeout=120)
            if proc.returncode != 0:
                print(f"[Engine] ffmpeg stderr: {stderr.decode(errors='replace')}")
        except Exception as e:
            print(f"[Engine] ffmpeg error: {e}")
        finally:
            if os.path.exists(wav_tmp):
                os.remove(wav_tmp)

        print(f"[Engine] Saved: {output_path}")
        return output_path


class PromptRewriter:
    """
    Loads a Qwen3 model for rewriting short prompts into high-quality
    Chinese dense captions optimized for NAVA inference.
    Supports GPU↔CPU offloading to share GPU with the NAVA backbone.

    IMPORTANT: Model loading is deferred to first rewrite() call so no
    CUDA operations happen at construction time (safe for ZeroGPU).
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.model_path = model_path or "Qwen/Qwen3-4B-Instruct-2507"
        self.tokenizer = None
        self.model = None
        self._initialized = False
        self._on_gpu = False
        self.system_prompt = SYSTEM_PROMPT
        print(f"[Rewriter] PromptRewriter created (deferred init). model={self.model_path}")

    def _lazy_init(self):
        """Load tokenizer and model on first use (inside @spaces.GPU scope)."""
        if self._initialized:
            return

        print(f"[Rewriter] Loading {self.model_path}...")
        t0 = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self._on_gpu = True
        self._initialized = True
        print(f"[Rewriter] Loaded in {time.time() - t0:.1f}s")

    def offload(self):
        """Move rewriter model to CPU to free GPU memory for inference."""
        if self._initialized and self._on_gpu:
            try:
                self.model.to("cpu")
            except Exception as e:
                print(f"[Rewriter] Note: could not manually offload model: {e}")
            torch.cuda.empty_cache()
            self._on_gpu = False
            print("[Rewriter] Offloaded to CPU")

    def reload(self):
        """Move rewriter model to cuda:0 for rewriting."""
        self._lazy_init()
        if not self._on_gpu:
            try:
                self.model.to("cuda:0")
            except Exception as e:
                print(f"[Rewriter] Note: could not manually reload model: {e}")
            self._on_gpu = True
            print("[Rewriter] Reloaded to cuda:0")

    def rewrite(self, user_input: str) -> tuple:
        """
        Rewrite prompt. Returns (result, warning) tuple.
        Warning is non-empty if <S><E> pair count mismatches.
        Must be called inside a @spaces.GPU decorated function on ZeroGPU.
        """
        self.reload()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"[Rewriter] Generating (input tokens: {inputs['input_ids'].shape[1]})...")
        t0 = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.3,
                top_p=0.75,
                top_k=20,
                do_sample=True,
                repetition_penalty=1.05,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Keep only content after the LAST </think> (discard all thinking blocks)
        if "</think>" in result:
            result = result.rsplit("</think>", 1)[-1].strip()
        # Strip any residual unclosed <think> block at the end
        if "<think>" in result:
            result = result.split("<think>", 1)[0].strip()

        elapsed = time.time() - t0
        print(f"[Rewriter] Done in {elapsed:.1f}s ({len(new_tokens)} tokens)")

        # Check <S><E> pair count
        input_count = _count_speech_tags(user_input)
        output_count = _count_speech_tags(result)
        warning = ""
        if input_count > 0 and output_count != input_count:
            warning = (
                f"⚠️ Speech tag count mismatch! Input has {input_count} <S><E> pairs, "
                f"output has {output_count} pairs. Please click Rewrite again."
            )
            print(f"[Rewriter] WARNING: {warning}")

        return result, warning


def worker_loop(engine: NAVAEngine):
    """Non-rank-0 processes wait for commands and execute inference."""
    import torch.distributed as dist
    rank = dist.get_rank()
    print(f"[Rank {rank}] Entering worker loop, waiting for commands...")

    # Trigger lazy init on workers too (inside this function which is
    # called after dist init, so CUDA is already initialized).
    engine._lazy_init()

    while True:
        cmd = broadcast_cmd(0, src=0)

        if cmd == CMD_EXIT:
            print(f"[Rank {rank}] Received EXIT command. Shutting down.")
            break
        elif cmd == CMD_INFER:
            # Receive all params from rank 0
            prompt = broadcast_string("", src=0)
            image_path = broadcast_string("", src=0)
            spk_wav_1 = broadcast_string("", src=0)
            spk_wav_2 = broadcast_string("", src=0)
            steps = broadcast_int(0, src=0)
            is_i2v = bool(broadcast_int(0, src=0))
            height = broadcast_int(0, src=0)
            width = broadcast_int(0, src=0)
            frames = broadcast_int(0, src=0)

            # Build spk_wav_paths
            spk_wav_paths = []
            if spk_wav_1:
                spk_wav_paths.append(spk_wav_1)
            if spk_wav_2:
                spk_wav_paths.append(spk_wav_2)

            # Run inference (result discarded on non-rank-0)
            engine.generate(
                prompt=prompt,
                image_path=image_path if image_path else None,
                spk_wav_paths=spk_wav_paths if spk_wav_paths else None,
                steps=steps,
                is_i2v=is_i2v,
                height=height,
                width=width,
                frames=frames,
            )


def run_gradio(engine: NAVAEngine, rewriter: PromptRewriter, args):
    """
    Build and launch the Gradio interface.
    In ZeroGPU mode: runs on rank 0 only; all CUDA ops are inside @spaces.GPU functions.
    In torchrun mode: runs on rank 0 only; other ranks are in worker_loop().
    """
    import gradio as gr

    # Determine if we need distributed broadcasting (torchrun SP mode only)
    _use_dist = engine.use_sp

    # ---- Callback: Rewrite ----
    @_spaces_gpu_decorator(size="xlarge")
    def rewrite_fn(user_prompt: str):
        """Rewrite prompt only, triggered by Rewrite button."""
        if not user_prompt.strip():
            return "", ""
        rewritten, warning = rewriter.rewrite(user_prompt)
        print(f"[Gradio] Rewritten prompt:\n{rewritten[:200]}...")
        return rewritten, warning

    # ---- Callback: Generate ----
    @_spaces_gpu_decorator(size="xlarge", duration=300)
    def infer_fn(user_prompt: str, rewritten_prompt: str, image_file: str,
                 spk_wav_1: str, spk_wav_2: str,
                 steps: int, duration_sec: int, aspect_ratio: str):
        """
        Main inference function triggered by Generate button.
        Uses rewritten_prompt if available, otherwise falls back to user_prompt.
        On ZeroGPU: runs entirely on the assigned GPU (no dist).
        On torchrun: broadcasts params to worker ranks before running.
        """
        # Convert duration (seconds) to frames: frames = 6 * seconds + 1
        frames = int(duration_sec) * 6 + 1

        # Use rewritten prompt if it exists, otherwise use raw input
        final_prompt = (
            rewritten_prompt.strip() if rewritten_prompt.strip() else user_prompt.strip()
        )

        # Resolve aspect ratio to height/width
        height, width = ASPECT_RATIO_MAP.get(aspect_ratio, (704, 1280))

        # I2V mode is automatically enabled when an image is provided
        is_i2v = bool(image_file)

        # Offload rewriter to free GPU memory for inference
        rewriter.offload()

        # Broadcast to worker ranks (SP/torchrun mode only)
        if _use_dist:
            broadcast_cmd(CMD_INFER, src=0)
            broadcast_string(final_prompt, src=0)
            broadcast_string(image_file or "", src=0)
            broadcast_string(spk_wav_1 or "", src=0)
            broadcast_string(spk_wav_2 or "", src=0)
            broadcast_int(steps, src=0)
            broadcast_int(int(is_i2v), src=0)
            broadcast_int(height, src=0)
            broadcast_int(width, src=0)
            broadcast_int(frames, src=0)

        # Build spk_wav_paths
        spk_wav_paths = []
        if spk_wav_1 and os.path.exists(spk_wav_1):
            spk_wav_paths.append(spk_wav_1)
        if spk_wav_2 and os.path.exists(spk_wav_2):
            spk_wav_paths.append(spk_wav_2)

        # Run inference on rank 0 (all ranks run in parallel via SP in torchrun mode)
        output_path = engine.generate(
            prompt=final_prompt,
            image_path=image_file if image_file else None,
            spk_wav_paths=spk_wav_paths if spk_wav_paths else None,
            steps=steps,
            is_i2v=is_i2v,
            height=height,
            width=width,
            frames=frames,
        )

        return output_path

    # ---- Custom CSS ----
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    .gr-button-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-secondary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4) !important;
    }
    #nava-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }
    #nava-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-top: 0 !important;
    }
    .tip-box {
        background: linear-gradient(135deg, #e0c3fc33 0%, #8ec5fc33 100%);
        border-left: 4px solid #764ba2;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    """

    # ---- Build Gradio Blocks ----
    with gr.Blocks(title="NAVA — Audio-Video Generator") as demo:

        # Header
        gr.HTML(
            """
            <div style="text-align:center; padding: 20px 0 8px;">
                <h1 id="nava-title">🎬 NAVA Audio-Video Generator</h1>
                <p id="nava-subtitle">
                    Native Audio-Visual Alignment • Prompt Rewrite
                </p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # ──── Left Column: Inputs ────
            with gr.Column(scale=2):

                gr.HTML(
                    """
                    <div class="tip-box">
                        <strong>⚡ Recommendation:</strong> For optimal generation quality,
                        use the <strong>Rewrite</strong> function — especially if your prompt
                        is in English or relatively brief. NAVA is primarily trained on
                        high-quality Chinese dense captions; the rewriter transforms your
                        input into the format that best activates the model's full potential.
                    </div>
                    """
                )

                # Prompt input
                with gr.Group():
                    prompt_input = gr.Textbox(
                        label="✏️ Prompt (— Step 1)",
                        placeholder=(
                            "Enter a short description or detailed prompt...\n"
                            "E.g.: A dragon breathing fire over a futuristic city at sunset"
                        ),
                        lines=4,
                        elem_id="prompt-input",
                    )

                    with gr.Row():                    
                        rewrite_btn = gr.Button(
                            "✨ Rewrite Prompt (— Step 2)", variant="secondary"
                        )

                    with gr.Row():   
                        rewritten_prompt = gr.Textbox(
                            label="📝 Rewritten Prompt (click Rewrite to generate, or use raw input)",
                            lines=8,
                            interactive=True,
                            elem_id="rewritten-prompt",
                        )
                    with gr.Row():  
                        speech_warning = gr.Textbox(
                            label="-> Speech Tag Check",
                            interactive=False,
                            visible=True,
                        )

                # Image input (optional, enables I2V)
                with gr.Accordion("🖼️ Image Input (optional — enables I2V mode)", open=False):
                    image_input = gr.Image(
                        label="First Frame Image",
                        type="filepath",
                    )

                # Speaker reference (optional)
                with gr.Accordion("🎤 Speaker Reference (optional, max 2)", open=False):
                    with gr.Row():
                        spk_wav_1_input = gr.Audio(
                            label="Speaker 1 WAV",
                            type="filepath",
                        )
                        spk_wav_2_input = gr.Audio(
                            label="Speaker 2 WAV",
                            type="filepath",
                        )

                # Generation parameters
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Settings (Recommended)")

                    steps_input = gr.Slider(
                        minimum=10, maximum=100, value=20,
                        step=5, label="Inference Steps",
                        info="More steps = better quality, slower generation",
                    )

                    duration_input = gr.Slider(
                        minimum=2, maximum=10, value=4,
                        step=1, label="Duration (seconds) — 6s = 37 frames",
                        info="Video length in seconds",
                    )

                    aspect_ratio_input = gr.Dropdown(
                        choices=list(ASPECT_RATIO_MAP.keys()),
                        value="1:1 (960×960)",
                        label="Aspect Ratio",
                    )

                submit_btn = gr.Button(
                    "🚀 Generate (— Step 3)", variant="primary", size="lg"
                )

            # ──── Right Column: Output ────
            with gr.Column(scale=2):
                video_output = gr.Video(
                    label="Generated Video (with Audio)(approx.300s)",
                    elem_id="video-output",
                    height=400,
                )

                gr.HTML(
                    """
                    <div style="text-align:center; padding:16px; color:#999; font-size:0.9em;">
                        <p>Generated videos include synchronized native audio.</p>
                        <p style="margin-top:4px;">
                            NAVA • 6.3B parameters • Native Audio-Visual Alignment • This is a demo Space, and more optimizations are coming soon
                        </p>
                    </div>
                    """
                )

        # ---- Event Wiring ----

        duration_input.change(
            fn=lambda s: gr.update(
                label=f"Duration (seconds) — {int(s)}s = {int(s) * 6 + 1} frames",
                minimum=2, maximum=10, step=1,
            ),
            inputs=[duration_input],
            outputs=[duration_input],
        )

        rewrite_btn.click(
            fn=rewrite_fn,
            inputs=[prompt_input],
            outputs=[rewritten_prompt, speech_warning],
        )

        submit_btn.click(
            fn=infer_fn,
            inputs=[
                prompt_input, rewritten_prompt, image_input,
                spk_wav_1_input, spk_wav_2_input,
                steps_input, duration_input, aspect_ratio_input,
            ],
            outputs=[video_output],
        )

    demo.queue(max_size=20)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, theme=orange_red_theme, css=custom_css)


def run_debug_gradio(args):
    """Launch Gradio in debug mode — no models loaded, UI-only for testing."""
    import gradio as gr

    def dummy_rewrite(user_prompt):
        """Simulate prompt rewriting."""
        time.sleep(0.5)
        return (
            f"[DEBUG REWRITE] 这是一段充满电影感与沉浸式氛围的视频。{user_prompt}。"
            f"画面中没有人物对白，也没有任何旁白。整体听感沉浸震撼，突出视觉冲击力。",
            "",
        )

    def dummy_infer(user_prompt, rewritten_prompt, image_file,
                    spk_wav_1, spk_wav_2, steps, duration_sec, aspect_ratio):
        """Simulate inference."""
        final = rewritten_prompt.strip() if rewritten_prompt.strip() else user_prompt
        height, width = ASPECT_RATIO_MAP.get(aspect_ratio, (704, 1280))
        frames = int(duration_sec) * 6 + 1
        is_i2v = bool(image_file)
        print(f"[DEBUG] Would generate with prompt: {final[:100]}...")
        print(f"[DEBUG] image={image_file}, spk1={spk_wav_1}, spk2={spk_wav_2}")
        print(f"[DEBUG] steps={steps}, frames={frames}, is_i2v={is_i2v}, {width}x{height}")
        return None

    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    .gr-button-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-secondary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.4) !important;
    }
    #nava-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }
    #nava-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-top: 0 !important;
    }
    .debug-banner {
        background: linear-gradient(135deg, #ff9a5633 0%, #ff614833 100%);
        border: 2px dashed #ff6148;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        color: #ff6148;
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 16px;
    }
    .tip-box {
        background: linear-gradient(135deg, #e0c3fc33 0%, #8ec5fc33 100%);
        border-left: 4px solid #764ba2;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    """

    with gr.Blocks(
        title="NAVA — Audio-Video Generator (DEBUG)",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.pink,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=custom_css,
    ) as demo:

        gr.HTML(
            """
            <div style="text-align:center; padding: 20px 0 8px;">
                <h1 id="nava-title">🎬 NAVA Audio-Video Generator</h1>
                <p id="nava-subtitle">
                    Native Audio-Visual Alignment • Debug Mode
                </p>
            </div>
            <div class="debug-banner">
                🛠️ DEBUG MODE — No models loaded, UI only. All actions are simulated.
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):

                gr.HTML(
                    """
                    <div class="tip-box">
                        <strong>⚡ Recommendation:</strong> For optimal generation quality,
                        use the <strong>Rewrite</strong> function — especially if your prompt
                        is in English or relatively brief.
                    </div>
                    """
                )

                with gr.Group():
                    prompt_input = gr.Textbox(
                        label="✏️ Prompt",
                        placeholder=(
                            "Enter a short description or detailed prompt...\n"
                            "E.g.: A dragon breathing fire over a futuristic city at sunset"
                        ),
                        lines=4,
                    )

                    rewrite_btn = gr.Button("✨ Rewrite Prompt", variant="secondary")

                    rewritten_prompt = gr.Textbox(
                        label="📝 Rewritten Prompt",
                        lines=8,
                        interactive=True,
                    )

                    speech_warning = gr.Textbox(
                        label="🔍 Speech Tag Check",
                        interactive=False,
                        visible=True,
                    )

                with gr.Accordion("🖼️ Image Input (optional — enables I2V mode)", open=False):
                    image_input = gr.Image(label="First Frame Image", type="filepath")

                with gr.Accordion("🎤 Speaker Reference (optional, max 2)", open=False):
                    with gr.Row():
                        spk_wav_1_input = gr.Audio(label="Speaker 1 WAV", type="filepath")
                        spk_wav_2_input = gr.Audio(label="Speaker 2 WAV", type="filepath")

                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Settings")

                    steps_input = gr.Slider(
                        minimum=10, maximum=100, value=25,
                        step=5, label="Inference Steps",
                        info="More steps = better quality, slower generation",
                    )

                    duration_input = gr.Slider(
                        minimum=2, maximum=10, value=3,
                        step=1, label="Duration (seconds) — 6s = 37 frames",
                        info="Video length in seconds",
                    )

                    aspect_ratio_input = gr.Dropdown(
                        choices=list(ASPECT_RATIO_MAP.keys()),
                        value="16:9 (1280×704)",
                        label="Aspect Ratio",
                    )

                submit_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                video_output = gr.Video(label="🎥 Generated Video (with Audio)", height=500)

                gr.HTML(
                    """
                    <div style="text-align:center; padding:16px; color:#999; font-size:0.9em;">
                        <p>Generated videos include synchronized native audio.</p>
                        <p style="margin-top:4px;">
                            NAVA • 6.3B parameters • Native Audio-Visual Alignment
                        </p>
                    </div>
                    """
                )

        # ---- Event Wiring ----
        duration_input.change(
            fn=lambda s: gr.update(
                label=f"Duration (seconds) — {int(s)}s = {int(s) * 6 + 1} frames",
                minimum=2, maximum=10, step=1,
            ),
            inputs=[duration_input],
            outputs=[duration_input],
        )

        rewrite_btn.click(
            fn=dummy_rewrite,
            inputs=[prompt_input],
            outputs=[rewritten_prompt, speech_warning],
        )

        submit_btn.click(
            fn=dummy_infer,
            inputs=[
                prompt_input, rewritten_prompt, image_input,
                spk_wav_1_input, spk_wav_2_input, steps_input,
                duration_input, aspect_ratio_input,
            ],
            outputs=[video_output],
        )

    demo.queue(max_size=1)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

def main():
    parser = argparse.ArgumentParser(
        description="NAVA — Single-file Gradio App (SP inference + prompt rewrite)"
    )
    parser.add_argument("--config", type=str, default="configs/nava.yaml",
                        help="NAVA config yaml path")
    parser.add_argument("--ckpt", type=str, default="NAVA.safetensors",
                        help="NAVA checkpoint path")
    parser.add_argument("--rewrite_model", type=str,
                        default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Rewrite model path")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--height", type=int, default=704,
                        help="Default video height")
    parser.add_argument("--width", type=int, default=1280,
                        help="Default video width")
    parser.add_argument("--frames", type=int, default=37,
                        help="Default number of video frames")
    parser.add_argument("--steps", type=int, default=50,
                        help="Default inference steps")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: skip all model loading, only launch Gradio UI")
    args = parser.parse_args()

    # ──── Debug mode: no models, no distributed, just UI ────
    if args.debug:
        print("=" * 60)
        print("  NAVA Gradio App — DEBUG MODE")
        print(f"  Port: {args.port}")
        print(f"  Share: {args.share}")
        print("=" * 60)
        run_debug_gradio(args)
        return

    # ──── Detect execution mode ────
    # torchrun sets RANK and WORLD_SIZE > 1 in the environment.
    # ZeroGPU / plain python sets neither (or WORLD_SIZE=1).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    _is_torchrun = world_size > 1

    if _is_torchrun:
        # ──── Multi-GPU torchrun / SP mode ────
        # Safe to touch CUDA here because torchrun does not use ZeroGPU emulation.
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=24),
        )

        print(f"[Rank {rank}] Initialized. device=cuda:{local_rank}, world_size={world_size}")

        engine = NAVAEngine(
            config_path=args.config,
            ckpt_path=args.ckpt,
            rank=rank,
            world_size=world_size,
            use_sp=True,
            height=args.height,
            width=args.width,
            frames=args.frames,
        )

        # All ranks barrier after object creation (before lazy init)
        dist.barrier()

        if rank == 0:
            try:
                ensure_weights()
            except Exception as e:
                print(f"[Startup] Error ensuring weights: {e}")
            rewriter = PromptRewriter(model_path=args.rewrite_model)
            run_gradio(engine, rewriter, args)
            # Tell workers to stop when Gradio exits
            broadcast_cmd(CMD_EXIT, src=0)
        else:
            worker_loop(engine)

        dist.barrier()
        dist.destroy_process_group()

    else:
        # ──── Single-GPU or ZeroGPU mode ────
        # DO NOT call torch.cuda.set_device() or dist.init_process_group() here.
        # All CUDA ops must happen inside @spaces.GPU decorated callbacks.
        print("=" * 60)
        print("  NAVA Gradio App — Single-GPU / ZeroGPU Mode")
        print(f"  ZeroGPU detected: {_IS_ZERO_GPU}")
        print(f"  spaces available: {_HAS_SPACES}")
        print(f"  Port: {args.port}  |  Share: {args.share}")
        print("=" * 60)

        if not args.config or not args.ckpt:
            print(
                "[Warning] --config and/or --ckpt not provided. "
                "Engine will fail on first generate() call. "
                "Use --debug for UI-only testing."
            )

        # Download weights on CPU before Gradio launches
        try:
            ensure_weights()
        except Exception as e:
            print(f"[Startup] Error ensuring weights: {e}")

        # Create engine and rewriter with fully deferred CUDA init.
        # No CUDA is touched here — safe for ZeroGPU startup.
        engine = NAVAEngine(
            config_path=args.config,
            ckpt_path=args.ckpt,
            rank=0,
            world_size=1,
            use_sp=False,   # No SP in single-GPU / ZeroGPU mode
            height=args.height,
            width=args.width,
            frames=args.frames,
        )

        rewriter = PromptRewriter(model_path=args.rewrite_model)

        run_gradio(engine, rewriter, args)


if __name__ == "__main__":
    main()