import time
import random
import requests
import streamlit as st

BASE_URL = "https://api.runcomfy.net/prod/v1"

def runcomfy_generate_image(
    api_key: str,
    deployment_id: str,
    prompt: str,
    negative: str,
    poll_interval: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str,
    scheduler: str,
):
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    nonce = int(time.time())
    seed = random.randint(1, 2**31 - 1)

    # ✅ 당신 workflow_api.json 기준 노드 ID
    payload = {
        "overrides": {
            "6": {"inputs": {"text": prompt}},  # positive
            "7": {"inputs": {"text": negative}},  # negative
            "3": {"inputs": {  # KSampler
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            }},
            "5": {"inputs": {"width": width, "height": height, "batch_size": 1}},
            "9": {"inputs": {"filename_prefix": f"ComfyUI_{nonce}"}},
        }
    }

    # 1) submit
    submit_res = requests.post(
        f"{BASE_URL}/deployments/{deployment_id}/inference",
        headers=headers,
        json=payload,
        timeout=60,
    )
    submit_res.raise_for_status()
    request_id = submit_res.json()["request_id"]

    # 2) poll
    while True:
        st_res = requests.get(
            f"{BASE_URL}/deployments/{deployment_id}/requests/{request_id}/status",
            headers=headers,
            timeout=60,
        )
        st_res.raise_for_status()
        status_data = st_res.json()
        status = (status_data.get("status") or "").lower()

        if status in ("succeeded", "completed"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"Run failed: {status_data}")

        time.sleep(poll_interval)

    # 3) result
    result_res = requests.get(
        f"{BASE_URL}/deployments/{deployment_id}/requests/{request_id}/result",
        headers=headers,
        timeout=60,
    )
    result_res.raise_for_status()
    result_data = result_res.json()

    # 4) parse output url (outputs is dict)
    image_url = None
    outputs = result_data.get("outputs", {})

    # Prefer SaveImage node "9"
    if isinstance(outputs, dict) and "9" in outputs:
        imgs = outputs["9"].get("images", [])
        if imgs:
            image_url = imgs[0].get("url")

    # Fallback: first node that has images
    if not image_url and isinstance(outputs, dict):
        for node_out in outputs.values():
            imgs = node_out.get("images", [])
            if imgs:
                image_url = imgs[0].get("url")
                break

    if not image_url:
        raise ValueError(f"이미지 URL을 찾지 못했습니다. keys={list(result_data.keys())}")

    return request_id, seed, image_url


st.set_page_config(page_title="RunComfy Generator", layout="centered")
st.title("RunComfy → Streamlit Cloud 이미지 생성")

# ✅ Streamlit Cloud에서는 secrets 사용 권장
api_key = st.secrets.get("RUNCOMFY_API_KEY", "")
deployment_id = st.secrets.get("RUNCOMFY_DEPLOYMENT_ID", "")

if not api_key or not deployment_id:
    st.error("Secrets에 RUNCOMFY_API_KEY / RUNCOMFY_DEPLOYMENT_ID를 설정해야 합니다.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    width = st.selectbox("Width", [512, 768, 1024], index=0)
    height = st.selectbox("Height", [512, 768, 1024], index=0)
    steps = st.slider("Steps", 10, 40, 20)
    cfg = st.slider("CFG", 1.0, 15.0, 8.0, step=0.5)
    denoise = st.slider("Denoise", 0.1, 1.0, 1.0, step=0.1)
    sampler_name = st.selectbox("Sampler", ["euler", "euler_a", "dpmpp_2m", "dpmpp_sde"], index=0)
    scheduler = st.selectbox("Scheduler", ["normal", "karras", "exponential"], index=0)
    poll_interval = st.selectbox("Poll interval (sec)", [1, 2, 3, 5], index=1)

prompt = st.text_area(
    "Prompt",
    value="cinematic film still, cyberpunk city, rain, neon lights, 8k, masterpiece",
    height=120,
)
negative = st.text_area("Negative Prompt", value="text, watermark", height=80)

if "busy" not in st.session_state:
    st.session_state.busy = False

generate = st.button("Generate", type="primary", use_container_width=True, disabled=st.session_state.busy)

if generate:
    st.session_state.busy = True
    try:
        status_box = st.empty()
        status_box.info("Submitting...")
        with st.spinner("Generating..."):
            request_id, seed, image_url = runcomfy_generate_image(
                api_key=api_key,
                deployment_id=deployment_id,
                prompt=prompt,
                negative=negative,
                poll_interval=poll_interval,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                denoise=denoise,
                sampler_name=sampler_name,
                scheduler=scheduler,
            )
        status_box.success(f"Done | request_id={request_id} | seed={seed}")
        st.image(image_url, caption=image_url, use_container_width=True)

    except Exception as e:
        st.error(f"Failed: {e}")
    finally:
        st.session_state.busy = False
