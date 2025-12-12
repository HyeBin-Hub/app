import os
import time
import random
import requests
import streamlit as st

BASE_URL = "https://api.runcomfy.net/prod/v1"

def runcomfy_generate_image(
    api_key: str,
    deployment_id: str,
    prompt: str,
    negative: str = "text, watermark, blurry, lowres",
    poll_interval: int = 2,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 8.0,
    denoise: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
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
            "6": {"inputs": {"text": prompt}},                 # Positive prompt
            "7": {"inputs": {"text": negative}},               # Negative prompt
            "3": {"inputs": {                                  # KSampler
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            }},
            "5": {"inputs": {"width": width, "height": height, "batch_size": 1}},  # Latent size
            "9": {"inputs": {"filename_prefix": f"ComfyUI_{nonce}"}},              # SaveImage prefix
        }
    }

    # 1) Submit
    submit_res = requests.post(
        f"{BASE_URL}/deployments/{deployment_id}/inference",
        headers=headers,
        json=payload,
        timeout=60,
    )
    submit_res.raise_for_status()
    request_id = submit_res.json()["request_id"]

    # 2) Poll status
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

    # 3) Get result
    result_res = requests.get(
        f"{BASE_URL}/deployments/{deployment_id}/requests/{request_id}/result",
        headers=headers,
        timeout=60,
    )
    result_res.raise_for_status()
    result_data = result_res.json()

    # 4) Parse image url (outputs는 dict)
    image_url = None
    outputs = result_data.get("outputs", {})

    if isinstance(outputs, dict) and "9" in outputs:
        imgs = outputs["9"].get("images", [])
        if imgs:
            image_url = imgs[0].get("url")

    if not image_url and isinstance(outputs, dict):
        for node_out in outputs.values():
            imgs = node_out.get("images", [])
            if imgs:
                image_url = imgs[0].get("url")
                break

    if not image_url:
        raise ValueError(f"이미지 URL을 찾지 못했습니다. keys={list(result_data.keys())}")

    return {
        "request_id": request_id,
        "seed": seed,
        "image_url": image_url,
        "status": status_data,
        "result": result_data,
    }


st.set_page_config(page_title="RunComfy Image Generator", layout="centered")
st.title("RunComfy → Streamlit 이미지 생성")

# ✅ 키/ID는 secrets 또는 환경변수로 받는 것을 권장
api_key = st.secrets.get("RUNCOMFY_API_KEY", os.getenv("RUNCOMFY_API_KEY", ""))
deployment_id = st.secrets.get("RUNCOMFY_DEPLOYMENT_ID", os.getenv("RUNCOMFY_DEPLOYMENT_ID", ""))

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("RUNCOMFY_API_KEY", type="password")
    if not deployment_id:
        deployment_id = st.text_input("RUNCOMFY_DEPLOYMENT_ID")

    width = st.selectbox("Width", [512, 768, 1024], index=0)
    height = st.selectbox("Height", [512, 768, 1024], index=0)
    steps = st.slider("Steps", 10, 40, 20)
    cfg = st.slider("CFG", 1.0, 15.0, 8.0, step=0.5)
    denoise = st.slider("Denoise", 0.1, 1.0, 1.0, step=0.1)
    poll_interval = st.selectbox("Poll interval (sec)", [1, 2, 3, 5], index=1)

prompt = st.text_area(
    "Prompt",
    value="cinematic film still, cyberpunk city, rain, neon lights, 8k, masterpiece",
    height=120
)
negative = st.text_area("Negative Prompt", value="text, watermark", height=80)

generate = st.button("Generate", type="primary", use_container_width=True)

if generate:
    if not api_key or not deployment_id:
        st.error("API Key와 Deployment ID를 입력하세요.")
        st.stop()

    status_box = st.empty()
    with st.spinner("RunComfy에 요청 중..."):
        try:
            # 진행 표시(간단 버전): 폴링 주기는 함수 내부에서 sleep
            status_box.info("Submitting job...")
            out = runcomfy_generate_image(
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
            )
            status_box.success(f"Done. request_id={out['request_id']} | seed={out['seed']}")
            st.image(out["image_url"], caption=out["image_url"], use_container_width=True)

        except Exception as e:
            status_box.error(f"Failed: {e}")
