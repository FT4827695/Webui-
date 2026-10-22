"""
图像编辑 - Forge Neo
"""
import os
import sys
import traceback
import urllib.parse
import datetime
import queue
import threading
import time
import subprocess
from pathlib import Path
from typing import List, Tuple
import gradio as gr
import numpy as np
import torch
from PIL import Image
from modules import images, shared, processing, devices, sd_models, scripts, paths
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.shared import opts
from modules.sd_samplers_common import images_tensor_to_samples

# 导入 Forge Neo 专用模块
try:
    from modules_forge import main_entry
    FORGE_MAIN_ENTRY_AVAILABLE = True
except ImportError:
    FORGE_MAIN_ENTRY_AVAILABLE = False

# 导入采样器和调度器模块
try:
    from modules import sd_samplers
    WEBUI_SAMPLERS_AVAILABLE = True
except ImportError:
    WEBUI_SAMPLERS_AVAILABLE = False

try:
    from modules import sd_schedulers
    WEBUI_SCHEDULERS_AVAILABLE = True
except ImportError:
    WEBUI_SCHEDULERS_AVAILABLE = False

# 定义后备输出目录（用于打开目录按钮）
OUTPUT_DIR = Path(shared.data_path) / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LoRA 模型目录
if hasattr(shared.cmd_opts, 'lora_dir') and shared.cmd_opts.lora_dir:
    LORA_PATH = Path(shared.cmd_opts.lora_dir)
else:
    LORA_PATH = Path(paths.models_path) / "Lora"
if not LORA_PATH.exists():
    alt_path = Path(paths.models_path) / "lora"
    if alt_path.exists():
        LORA_PATH = alt_path
LORA_PATH.mkdir(parents=True, exist_ok=True)

# ==================== LoRA 模型列表获取 ====================
def get_lora_choices():
    try:
        if not LORA_PATH.exists():
            LORA_PATH.mkdir(parents=True, exist_ok=True)
            return [("无", " ")]
        lora_files = list(LORA_PATH.glob("*.safetensors"))
        lora_files.extend(LORA_PATH.glob("*.ckpt"))
        lora_files.extend(LORA_PATH.rglob("*.pt"))
        lora_files.extend(LORA_PATH.rglob("*.bin"))
        choices = [("无", " ")]
        for f in sorted(lora_files):
            rel_path = f.relative_to(LORA_PATH)
            display_name = str(rel_path.with_suffix(''))
            choices.append((display_name, str(f)))
        return choices
    except Exception as e:
        print(f"获取LoRA模型列表时出错: {e}")
        return [("无", " ")]

def get_lora_names():
    try:
        if not LORA_PATH.exists():
            return []
        lora_files = list(LORA_PATH.glob("*.safetensors"))
        lora_files.extend(LORA_PATH.glob("*.ckpt"))
        return [f.name for f in sorted(lora_files)]
    except Exception as e:
        print(f"获取LoRA名称列表时出错: {e}")
        return []

def refresh_lora_models():
    choices = get_lora_choices()
    return [gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)]

# ==================== 镜头角度选择器 HTML 路径 ====================
def get_angle_html_path():
    return str(Path(__file__).parent / "camera_3d_view.html")

# ==================== Reset_references ====================
def reset_references(p: StableDiffusionProcessingTxt2Img):
    try:
        p.clear_prompt_cache()
        if hasattr(p.sd_model, 'clear_references'):
            p.sd_model.clear_references()
    except Exception:
        pass

# ==================== 内嵌参考图编码函数 ====================
def encode_reference_images(p, references):
    if not references or p.sd_model is None:
        return
    try:
        from backend.args import dynamic_args
    except ImportError:
        class DynamicArgs:
            is_referencing = False
            kontext = False
            edit = False
            klein = False
        dynamic_args = DynamicArgs()
    dynamic_args.is_referencing = True
    dynamic_args.kontext = True
    try:
        for ref in references:
            if ref is None:
                continue
            w, h = ref.size
            if w % 64 != 0 or h % 64 != 0:
                new_w = max(64, round(w / 64) * 64)
                new_h = max(64, round(h / 64) * 64)
                ref = images.resize_image(1, ref, new_w, new_h)
            image = images.flatten(ref, opts.img2img_background_color)
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            image = torch.from_numpy(image).to(device=devices.device, dtype=torch.float32)
            images_tensor_to_samples(image.unsqueeze(0), 0, p.sd_model)
    finally:
        dynamic_args.is_referencing = False
        dynamic_args.kontext = False

# ==================== 使用 Forge Neo API 加载模型 ====================
def force_load_model():
    print("[Image Edit] === 开始加载模型（Forge Neo API） ===")
    try:
        checkpoint_name = opts.sd_model_checkpoint
        if not checkpoint_name:
            return "错误：请先在设置中选择模型"
        print(f"[Image Edit] 目标模型: {checkpoint_name}")
        
        from modules import sd_models
        if hasattr(sd_models, 'model_data'):
            if hasattr(sd_models.model_data, 'loaded_model'):
                sd_models.model_data.loaded_model = None
            if hasattr(sd_models.model_data, 'current_checkpoint'):
                sd_models.model_data.current_checkpoint = None
        if hasattr(sd_models, 'checkpoints_loaded'):
            sd_models.checkpoints_loaded.clear()
            
        if FORGE_MAIN_ENTRY_AVAILABLE:
            main_entry.refresh_model_loading_parameters(refresh=True)
        else:
            from modules.sd_models import select_checkpoint
            checkpoint_info = select_checkpoint()
            if checkpoint_info is None:
                return "错误：未找到模型文件"
            from modules_forge.presets import forge_unet_storage_dtype_options
            forge_unet_dtype = getattr(opts, 'forge_unet_storage_dtype', None)
            unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(forge_unet_dtype, (None, False))
            sd_models.model_data.forge_loading_parameters = dict(
                checkpoint_info=checkpoint_info,
                additional_modules=getattr(opts, 'forge_additional_modules', None),
                unet_storage_dtype=unet_storage_dtype
            )
            from backend.args import dynamic_args
            dynamic_args.online_lora = lora_fp16
            
        processing.need_global_unload = True
        sd_model, reloaded = sd_models.forge_model_reload()
        if sd_model is None:
            print("[Image Edit] ⚠️ forge_model_reload 返回空，尝试使用原生 load_model 兜底...")
            if hasattr(sd_models, 'load_model'):
                sd_models.load_model()
                sd_model = shared.sd_model
                
        if sd_model is None:
            return "错误：模型加载失败，请检查模型文件"
            
        shared.sd_model = sd_model
        print(f"[Image Edit] 模型加载成功: {sd_model.__class__.__name__}")
        model_name = getattr(sd_model, 'sd_checkpoint_info', None)
        model_name = model_name.model_name if model_name else checkpoint_name
        return f"成功加载模型: {model_name}"
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def unload_model() -> str:
    if shared.sd_model is None:
        return "模型已经卸载，无需重复操作"
    try:
        if hasattr(shared.sd_model, 'clear_references'):
            shared.sd_model.clear_references()
            
        from modules import sd_models
        if hasattr(sd_models, 'unload_model_weights'):
            sd_models.unload_model_weights()
        else:
            shared.sd_model = None
            
        if hasattr(sd_models, 'model_data'):
            if hasattr(sd_models.model_data, 'loaded_model'):
                sd_models.model_data.loaded_model = None
            if hasattr(sd_models.model_data, 'current_checkpoint'):
                sd_models.model_data.current_checkpoint = None
        if hasattr(sd_models, 'checkpoints_loaded'):
            sd_models.checkpoints_loaded.clear()
            
        processing.need_global_unload = True
        devices.torch_gc()
        print("[Image Edit] 模型已卸载")
        return "模型已卸载"
    except Exception as e:
        print(f"[Image Edit] ❌ 卸载失败: {e}")
        return f"❌ 卸载失败: {str(e)}"

def get_current_model_name() -> str:
    if shared.sd_model and not shared.sd_model.__class__.__name__.startswith('Fake'):
        if hasattr(shared.sd_model, 'sd_checkpoint_info'):
            return shared.sd_model.sd_checkpoint_info.model_name
        return opts.sd_model_checkpoint or "已加载模型"
    return "未激活（请点击「加载模型」）"

def refresh_display() -> str:
    return get_current_model_name()

def extract_images_from_uploads(img1, img2, img3) -> List[Image.Image]:
    images_list = []
    for img in [img1, img2, img3]:
        if img is not None:
            if isinstance(img, str):
                try:
                    img = Image.open(img)
                except Exception:
                    continue
            if isinstance(img, Image.Image):
                images_list.append(img)
            elif isinstance(img, np.ndarray):
                images_list.append(Image.fromarray(img))
    return images_list

def format_parameters(prompt, neg_prompt, width, height, steps, cfg, sampler_name, scheduler_name, seed, lora1, lora2, lora3, weight1, weight2, weight3, ref_count, num_images):
    lines = [
        f"Steps: {steps}",
        f"Sampler: {sampler_name}",
        f"Scheduler: {scheduler_name}",
        f"CFG scale: {cfg}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
        f"Model: {opts.sd_model_checkpoint or '未知'}",
        f"参考图数量: {ref_count}",
        f"生成数量: {num_images}",
        f"LoRA 1: {lora1} (权重: {weight1})" if lora1 and lora1 != "无" else "LoRA 1: 无",
        f"LoRA 2: {lora2} (权重: {weight2})" if lora2 and lora2 != "无" else "LoRA 2: 无",
        f"LoRA 3: {lora3} (权重: {weight3})" if lora3 and lora3 != "无" else "LoRA 3: 无",
        f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        f"Negative prompt: {neg_prompt[:100]}{'...' if len(neg_prompt) > 100 else ''}"
    ]
    return "\n".join(lines)

def generate_with_references(
    prompt: str,
    negative_prompt: str,
    img1,
    img2,
    img3,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    scheduler_name: str,
    seed: int,
    lora1_path: str,
    lora2_path: str,
    lora3_path: str,
    lora1_weight: float,
    lora2_weight: float,
    lora3_weight: float,
    num_images: int,
    progress=gr.Progress(),
) -> Tuple[List[Image.Image], str]:
    start_time = time.time()
    
    if not prompt.strip():
        return [], "提示词不能为空"
    if shared.sd_model is None or shared.sd_model.__class__.__name__.startswith('Fake'):
        return [], "模型未加载。请先点击「加载模型」按钮。"
    if not hasattr(shared.sd_model, 'encode_first_stage'):
        return [], "当前模型不支持图像编辑功能。请使用 Flux-Kontext / Qwen-Image-Edit 等模型。"
        
    references = extract_images_from_uploads(img1, img2, img3)
    if not references:
        return [], "请至少上传一张参考图"
        
    lora_prompt = ""
    if lora1_path and lora1_path != "无":
        lora_name = os.path.splitext(os.path.basename(lora1_path))[0]
        lora_prompt += f"<lora:{lora_name}:{lora1_weight}>"
    if lora2_path and lora2_path != "无":
        lora_name2 = os.path.splitext(os.path.basename(lora2_path))[0]
        lora_prompt += f" <lora:{lora_name2}:{lora2_weight}>"
    if lora3_path and lora3_path != "无":
        lora_name3 = os.path.splitext(os.path.basename(lora3_path))[0]
        lora_prompt += f" <lora:{lora_name3}:{lora3_weight}>"
        
    if lora_prompt:
        prompt = f"{prompt} {lora_prompt}".strip()
        print(f"[Image Edit] 已加载 LoRA: {lora_prompt}")
        
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_txt2img_samples or opts.outdir_samples or "",
        outpath_grids=opts.outdir_txt2img_grids or opts.outdir_grids or "",
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        width=width,
        height=height,
        batch_size=1,
        n_iter=num_images,
        seed=seed,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
    )
    reset_references(p)
    encode_reference_images(p, references)
    
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def run_generation():
        try:
            processed = process_images(p)
            result_queue.put(processed)
        except Exception as e:
            error_queue.put(str(e))
            
    thread = threading.Thread(target=run_generation)
    thread.start()
    
    total_steps = steps
    progress(0, desc="开始生成...")
    last_step = -1
    while thread.is_alive():
        current_step = shared.state.sampling_step
        if current_step != last_step:
            last_step = current_step
            if total_steps > 0:
                progress(current_step / total_steps, desc=f"步骤 {current_step}/{total_steps}")
            else:
                progress(0.5, desc=f"步骤 {current_step}")
        time.sleep(0.3)
        
    if not error_queue.empty():
        error_msg = error_queue.get()
        if "NaN" in error_msg:
            error_msg += "\n\n提示：检测到 NaN。请检查采样器与调度器类型"
        return [], error_msg
        
    if result_queue.empty():
        return [], "⚠️ 生成完成但未返回结果"
        
    processed = result_queue.get()
    if not processed or not processed.images:
        return [], "生成失败或未生成图像"
    progress(1.0, desc="生成完成！")
    
    param_str = format_parameters(
        prompt, negative_prompt, width, height, steps, cfg_scale,
        sampler_name, scheduler_name, seed,
        lora1_path, lora2_path, lora3_path,
        lora1_weight, lora2_weight, lora3_weight,
        len(references),
        num_images
    )
    
    elapsed = time.time() - start_time
    if elapsed < 60:
        time_str = f"{elapsed:.1f} 秒"
    else:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes} 分 {seconds} 秒"
    param_str += f"\n耗时: {time_str}"
    
    reset_references(p)
    return processed.images, f"成功生成 {len(processed.images)} 张图像\n\n{param_str}"

def create_ui():
    if WEBUI_SAMPLERS_AVAILABLE:
        try:
            sampler_choices = [(sampler.name, sampler.name) for sampler in sd_samplers.visible_samplers()]
        except:
            sampler_choices = [
                ("Euler", "Euler"),
                ("Euler Ancestral", "Euler Ancestral"),
                ("Heun", "Heun"),
                ("DPM++ 2M", "DPM++ 2M")
            ]
    else:
        sampler_choices = [
            ("Euler", "Euler"),
            ("Euler Ancestral", "Euler Ancestral"),
            ("Heun", "Heun"),
            ("DPM++ 2M", "DPM++ 2M")
        ]

    if WEBUI_SCHEDULERS_AVAILABLE:
        try:
            scheduler_choices = [(scheduler.label, scheduler.name) for scheduler in sd_schedulers.schedulers]
        except:
            scheduler_choices = [
                ("Automatic", "automatic"),
                ("Karras", "karras"),
                ("Exponential", "exponential"),
                ("SGM Uniform", "sgm_uniform"),
                ("Simple", "simple"),
                ("Normal", "normal"),
                ("DDIM", "ddim_uniform")
            ]
    else:
        scheduler_choices = [
            ("Automatic", "automatic"),
            ("Karras", "karras"),
            ("Exponential", "exponential"),
            ("SGM Uniform", "sgm_uniform"),
            ("Simple", "simple"),
            ("Normal", "normal"),
            ("DDIM", "ddim_uniform")
        ]
        
    lora_choices = get_lora_choices()
    angle_html_path = get_angle_html_path()
    
    if not os.path.exists(angle_html_path):
        print(f"[Image Edit] 警告: 镜头角度选择器 HTML 文件不存在: {angle_html_path}")
        
    apply_angle_js = """
    async () => {
        const iframe = document.getElementById('camera-angle-iframe');
        if (!iframe || !iframe.contentWindow) {
            alert('错误：无法访问3D视角界面，请刷新页面后重试！');
            return null;
        }

        await new Promise(resolve => {
            if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                resolve();
            } else {
                iframe.onload = resolve;
                setTimeout(resolve, 1500);
            }
        });

        iframe.contentWindow.postMessage({ type: 'GET_CURRENT_ANGLE' }, window.location.origin);

        return new Promise((resolve) => {
            let timeoutId = null;

            const matchText = (val, rules) => {
                const v = String(val ?? '').trim().toLowerCase();
                if (!v) {
                    return { matched: true, text: '' };
                }

                for (const r of rules) {
                    for (const k of r.keys) {
                        if (v.includes(k)) {
                            return { matched: true, text: r.text };
                        }
                    }
                }

                return { matched: false, text: '' };
            };

            const azimuthRules = [
                { keys: ['front-right', '前右'], text: '将镜头向右旋转45度' },
                { keys: ['right side', '右侧'], text: '将镜头向右旋转90度' },
                { keys: ['back-right', '后右'], text: '将镜头向右旋转135度' },
                { keys: ['back-left', '后左'], text: '将镜头向左旋转135度' },
                { keys: ['left side', '左侧'], text: '将镜头向左旋转90度' },
                { keys: ['front-left', '前左'], text: '将镜头向左旋转45度' },
                { keys: ['back view', 'back', '背面'], text: '将镜头旋转180度' },
                { keys: ['front view', 'front', '正面'], text: '' }
            ];

            const elevationRules = [
                { keys: ['low-angle', 'low', '低角度', '低'], text: '将镜头向下移动' },
                { keys: ['elevated', '仰拍', '仰'], text: '将镜头向上倾斜' },
                { keys: ['high-angle', 'high', 'top-down', '俯', '高角度', '高'], text: '将镜头转为俯视' },
                { keys: ['eye-level', 'eye', '平视'], text: '' }
            ];

            const distanceRules = [
                { keys: ['close-up', 'close', '特写'], text: '将镜头转为特写镜头' },
                { keys: ['medium', '中景'], text: '中景' },
                { keys: ['wide-angle', 'wide', '远景', '广角'], text: '将镜头转为广角镜头' }
            ];

            const handleMessage = (event) => {
                if (event.data.type !== 'ANGLE_SELECTED') {
                    return;
                }

                clearTimeout(timeoutId);
                window.removeEventListener('message', handleMessage);

                const raw = event.data;

                const azimuth = String(raw.azimuth ?? '').trim();
                const elevation = String(raw.elevation ?? '').trim();
                const distance = String(raw.distance ?? '').trim();

                const app = typeof gradioApp === 'function' ? gradioApp() : document;

                let versionText = '';
                app.querySelectorAll('#angle_model_version input').forEach(inp => {
                    versionText += (inp.value || '') + ' ';
                });

                let is2509 = false;

                if (versionText.includes('2509')) {
                    is2509 = true;
                } else if (versionText.includes('2511')) {
                    is2509 = false;
                } else {
                    const modelNameEl = app.querySelector('#model_status_box input, #model_status_box textarea');
                    const modelName = (modelNameEl ? modelNameEl.value : '').toLowerCase();
                    is2509 = modelName.includes('2509');
                }

                let anglePrompt = '';

                if (is2509) {
                    const azRes = matchText(azimuth, azimuthRules);
                    const elRes = matchText(elevation, elevationRules);
                    const distRes = matchText(distance, distanceRules);

                    if (!azRes.matched || !elRes.matched || !distRes.matched) {
                        alert(
                            '2509 未匹配到角度词。收到：方位=' +
                            (azimuth || '空') +
                            '，高程=' +
                            (elevation || '空') +
                            '，距离=' +
                            (distance || '空')
                        );
                        resolve(null);
                        return;
                    }

                    const parts2509 = [azRes.text, elRes.text, distRes.text].filter(Boolean);

                    if (parts2509.length === 0) {
                        resolve(null);
                        return;
                    }

                    anglePrompt = parts2509.join(' ');

                } else {
                    const parts = [];

                    if (raw.azimuth != null && raw.azimuth !== '') parts.push(raw.azimuth);
                    if (raw.elevation != null && raw.elevation !== '') parts.push(raw.elevation);
                    if (raw.distance != null && raw.distance !== '') parts.push(raw.distance);

                    if (parts.length === 0) {
                        alert('错误：未能获取到有效的角度数据！');
                        resolve(null);
                        return;
                    }

                    anglePrompt = '<sks> ' + parts.join(' ');
                }

                const promptContainer = document.getElementById('image_edit_prompt');
                if (promptContainer) {
                    let textarea = promptContainer.querySelector('textarea');

                    if (!textarea && (promptContainer.tagName === 'TEXTAREA' || promptContainer.tagName === 'INPUT')) {
                        textarea = promptContainer;
                    }

                    if (textarea) {
                        const newValue = textarea.value ? textarea.value + ' ' + anglePrompt : anglePrompt;

                        textarea.focus();
                        textarea.value = newValue;
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                        textarea.dispatchEvent(new Event('change', { bubbles: true }));
                        textarea.blur();

                        resolve(anglePrompt);
                        return;
                    }
                }

                alert('未找到正向提示词输入框！请检查 elem_id 是否为 image_edit_prompt');
                resolve(null);
            };

            timeoutId = setTimeout(() => {
                window.removeEventListener('message', handleMessage);
                alert('错误：等待3D视角界面响应超时！');
                resolve(null);
            }, 5000);

            window.addEventListener('message', handleMessage);
        });
    }
    """
    with gr.Blocks(css="""
        .model-status-textbox { width: 300px !important; min-width: 200px !important; max-width: 400px !important; }
        .align-btn { height: 38px !important; line-height: 38px !important; padding: 0 12px !important; }
        .angle-iframe-container { width: 100%; height: 350px; border: 1px solid #444; border-radius: 8px; overflow: hidden; }
    """) as ui:
        gr.Markdown("## 🖼️ 图像编辑")
        with gr.Tabs():
            with gr.TabItem("图像编辑", id="image_edit_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_status = gr.Textbox(label="模型状态", value=get_current_model_name(), interactive=False, elem_id="model_status_box", elem_classes="model-status-textbox")
                        with gr.Row():
                            force_load_btn = gr.Button("⚡ 加载模型", elem_classes="align-btn")
                            refresh_btn = gr.Button("🔄 卸载模型", elem_classes="align-btn")
                            reset_lora_btn = gr.Button("🔄 重置 LoRA", elem_classes="align-btn")
                        with gr.Row():
                            edit_image1 = gr.Image(type="pil", label="参考图像 1", interactive=True, height=300)
                            edit_image2 = gr.Image(type="pil", label="参考图像 2", interactive=True, height=300)
                            edit_image3 = gr.Image(type="pil", label="参考图像 3", interactive=True, height=300)
                        prompt = gr.Textbox(label="正向提示词", lines=3, elem_id="image_edit_prompt")
                        neg_prompt = gr.Textbox(label="负向提示词", lines=2)
                        with gr.Row():
                            width = gr.Slider(256, 2048, 1024, step=64, label="宽度")
                            height = gr.Slider(256, 2048, 1024, step=64, label="高度")
                        with gr.Row():
                            steps = gr.Slider(1, 50, 10, step=1, label="步数")
                            cfg = gr.Slider(1.0, 30.0, 1.0, step=0.5, label="CFG")
                        with gr.Row():
                            sampler = gr.Dropdown(choices=sampler_choices, value=sampler_choices[0][1] if sampler_choices else "Euler", label="采样器")
                            scheduler = gr.Dropdown(choices=scheduler_choices, value=scheduler_choices[0][1] if scheduler_choices else "automatic", label="调度器")
                        with gr.Row():
                            seed = gr.Number(-1, label="随机种子", precision=0)
                        num_images = gr.Slider(minimum=1, maximum=6, step=1, value=1, label="生成数量（单次多张）")
                        
                        with gr.Accordion("🎥 镜头角度选择器", open=False):
                            if os.path.exists(angle_html_path):
                                gr.HTML(f'''<div style="width: 100%; height: 350px; border: 1px solid #444; border-radius: 8px; overflow: hidden;"><iframe id="camera-angle-iframe" src="/file={urllib.parse.quote(angle_html_path)}" width="100%" height="100%" style="border: none;"></iframe></div>''')
                                model_version_dropdown = gr.Dropdown(
                                    choices=["自动检测 (推荐)", "Qwen-Edit-2511", "Qwen-Edit-2509"],
                                    value="自动检测 (推荐)",
                                    label="目标模型版本",
                                    elem_id="angle_model_version",
                                    interactive=True
                                )
                                angle_apply_btn = gr.Button("应用到正向提示词", variant="secondary", elem_classes="align-btn")
                                angle_apply_btn.click(None, [], [], _js=apply_angle_js)
                            else:
                                gr.Markdown(f"⚠️ 镜头角度选择器文件未找到：{angle_html_path}")
                                
                        with gr.Accordion("LoRA 模型快捷选择", open=False):
                            with gr.Row():
                                lora_dropdown1 = gr.Dropdown(choices=lora_choices, value="无", label="LoRA模型1", interactive=True, scale=3)
                                lora_weight1 = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="权重1", scale=1)
                            with gr.Row():
                                lora_dropdown2 = gr.Dropdown(choices=lora_choices, value="无", label="LoRA模型2", interactive=True, scale=3)
                                lora_weight2 = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="权重2", scale=1)
                            with gr.Row():
                                lora_dropdown3 = gr.Dropdown(choices=lora_choices, value="无", label="LoRA模型3", interactive=True, scale=3)
                                lora_weight3 = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="权重3", scale=1)
                            with gr.Row():
                                lora_refresh_btn = gr.Button("刷新LoRA模型列表", variant="secondary", size="sm")
                                lora_refresh_btn.click(fn=refresh_lora_models, inputs=[], outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3])
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery(label="生成结果", columns=2, rows=2, height=400)
                        with gr.Row():
                            generate_btn = gr.Button("🚀 生成图像", variant="primary", scale=3)
                            stop_btn = gr.Button("⏹️ 终止", variant="stop", scale=1)
                            clear_gallery_btn = gr.Button("🗑️ 清除图像", variant="secondary", scale=1)
                        status = gr.Textbox(label="状态", lines=8, interactive=False)
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空参考图", variant="secondary", scale=1)
                            open_output_btn = gr.Button("📂 打开输出目录", variant="secondary", scale=1)
                            
                state = gr.State(value="")
                
                def reset_lora_selections():
                    return (gr.update(value=""), gr.update(value=""), gr.update(value=""),
                            gr.update(value=1.0), gr.update(value=1.0), gr.update(value=1.0))
                reset_lora_btn.click(fn=reset_lora_selections, inputs=[], outputs=[lora_dropdown1, lora_dropdown2, lora_dropdown3, lora_weight1, lora_weight2, lora_weight3])
                
                force_load_btn.click(fn=force_load_model, outputs=[model_status]).then(fn=refresh_display, outputs=[model_status])
                refresh_btn.click(fn=unload_model, outputs=[model_status])
                
                clear_btn.click(fn=lambda: [None, None, None], outputs=[edit_image1, edit_image2, edit_image3])
                
                def open_output_directory():
                    base_dir = opts.outdir_txt2img_samples or opts.outdir_samples or str(OUTPUT_DIR)
                    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    target_dir = os.path.join(base_dir, today_str)
                    if not os.path.exists(target_dir):
                        target_dir = base_dir
                    if sys.platform == "win32":
                        os.startfile(target_dir)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", target_dir])
                    else:
                        subprocess.Popen(["xdg-open", target_dir])
                    return f"已打开输出目录: {target_dir}"
                open_output_btn.click(fn=open_output_directory, inputs=[], outputs=[])
                
                def set_status_loading():
                    return gr.update(value="⏳ 生成中...")
                def clear_gallery():
                    return []   
                    
                generate_btn.click(
                    fn=clear_gallery,
                    inputs=[],
                    outputs=[output_gallery]
                ).then(
                    fn=set_status_loading,
                    inputs=[],
                    outputs=[status]
                ).then(
                    fn=generate_with_references,
                    inputs=[
                        prompt, neg_prompt,
                        edit_image1, edit_image2, edit_image3,
                        width, height, steps, cfg,
                        sampler, scheduler, seed,
                        lora_dropdown1, lora_dropdown2, lora_dropdown3,
                        lora_weight1, lora_weight2, lora_weight3,
                        num_images,
                    ],
                    outputs=[output_gallery, state],
                    show_progress="full"   
                )
                
                def update_status_from_state(state_text):
                    return gr.update(value=state_text)
                state.change(
                    fn=update_status_from_state,
                    inputs=[state],
                    outputs=[status],
                    show_progress="hidden"   
                )
                
                stop_btn.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
                
                clear_gallery_btn.click(fn=clear_gallery, inputs=[], outputs=[output_gallery])
                
            with gr.TabItem("LoRA", id="lora_manager_tab"):
                gr.Markdown("### 选择要加载的 LoRA 模型（默认权重为 1.0）")
                gr.Markdown("勾选一个或多个 LoRA，点击「生成 LoRA 提示词」，将生成的文本发送到正向提示词框中。")
                lora_names = get_lora_names()
                lora_checkbox_group = gr.CheckboxGroup(
                    choices=lora_names,
                    label="LoRA 模型列表",
                    interactive=True,
                    info="可多选"
                )
                generate_lora_prompt_btn = gr.Button("📋 生成 LoRA 提示词", variant="primary")
                lora_prompt_output = gr.Textbox(
                    label="生成的 LoRA 提示词（可编辑）",
                    lines=3,
                    interactive=True,
                    elem_id="lora_prompt_output_textbox",
                    info="可手动删减或修改，然后发送到正向提示词"
                )
                send_to_prompt_btn = gr.Button("📤 发送到正向提示词", variant="secondary", size="sm")
                refresh_lora_list_btn = gr.Button("🔄 刷新 LoRA 列表", variant="secondary")
                
                def refresh_lora_list():
                    names = get_lora_names()
                    return gr.update(choices=names)
                refresh_lora_list_btn.click(fn=refresh_lora_list, inputs=[], outputs=[lora_checkbox_group])
                
                def make_lora_prompt(selected_loras):
                    if not selected_loras:
                        return "请至少选择一个 LoRA 模型"
                    prompts = []
                    for lora_name in selected_loras:
                        name_without_ext = os.path.splitext(lora_name)[0]
                        prompts.append(f"<lora:{name_without_ext}:1.0>")
                    return " ".join(prompts)
                    
                generate_lora_prompt_btn.click(fn=make_lora_prompt, inputs=[lora_checkbox_group], outputs=[lora_prompt_output])
                
                send_to_prompt_js = """
                async () => {
                    console.log("发送 LoRA 提示词：开始");
                    let container = document.getElementById('lora_prompt_output_textbox');
                    if (!container) {
                        const root = window.gradioApp ? window.gradioApp() : document;
                        container = root.querySelector('#lora_prompt_output_textbox');
                    }
                    if (!container) {
                        alert('未找到 LoRA 提示词文本框容器 (lora_prompt_output_textbox)');
                        return;
                    }
                    let loraInput = container.querySelector('textarea, input[type="text"]');
                    if (!loraInput && (container.tagName === 'TEXTAREA' || container.tagName === 'INPUT')) {
                        loraInput = container;
                    }
                    if (!loraInput) {
                        alert('无法找到 LoRA 提示词的输入框');
                        return;
                    }
                    const loraPrompt = loraInput.value;
                    if (!loraPrompt || !loraPrompt.trim()) {
                        alert('当前 LoRA 提示词为空，请先生成或编辑提示词');
                        return;
                    }
                    let promptContainer = document.getElementById('image_edit_prompt');
                    if (!promptContainer) {
                        const root = window.gradioApp ? window.gradioApp() : document;
                        promptContainer = root.querySelector('#image_edit_prompt');
                    }
                    if (!promptContainer) {
                        alert('未找到正向提示词容器 (image_edit_prompt)');
                        return;
                    }
                    let promptInput = promptContainer.querySelector('textarea, input[type="text"]');
                    if (!promptInput && (promptContainer.tagName === 'TEXTAREA' || promptContainer.tagName === 'INPUT')) {
                        promptInput = promptContainer;
                    }
                    if (!promptInput) {
                        alert('未找到正向提示词输入框');
                        return;
                    }
                    const originalValue = promptInput.value;
                    const newValue = originalValue.trim() ? originalValue + ' ' + loraPrompt : loraPrompt;
                    promptInput.focus();
                    promptInput.value = newValue;
                    promptInput.dispatchEvent(new Event('input', { bubbles: true }));
                    promptInput.dispatchEvent(new Event('change', { bubbles: true }));
                    promptInput.blur();
                }
                """
                send_to_prompt_btn.click(None, [], [], _js=send_to_prompt_js)
                ui.load(fn=refresh_lora_list, outputs=[lora_checkbox_group])
                
        ui.load(fn=refresh_display, outputs=[model_status])
        
    return ui

def on_ui_tabs():
    return [(create_ui(), "图像编辑", "image_edit")]

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_ui_tabs(on_ui_tabs)
    print("[Image Edit] 图像编辑已加载")
except Exception as e:
    print(f"[Image Edit] 注册失败: {e}")
