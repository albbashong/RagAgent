import threading
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


class LLMManager:
    def __init__(self, model_name="Qwen/Qwen3-1.7B", prompt_path: Path = Path("managers/prompt_config.yaml")):
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.prompts = self._load_prompts()
        self._prompts_mtime = self._get_prompt_mtime()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self._lock = threading.Lock()
        print(f"[LLM] ✅ {model_name} loaded with {len(self.prompts)} prompt profiles")

    # --------------------------------------------------------------
    # YAML prompt 로드
    # --------------------------------------------------------------
    def _load_prompts(self) -> dict:
        if not Path(self.prompt_path).exists():
            print(f"[LLM] ⚠️ Prompt config not found at {self.prompt_path}")
            return {}
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_prompt_mtime(self) -> float | None:
        try:
            return Path(self.prompt_path).stat().st_mtime
        except FileNotFoundError:
            return None

    def _maybe_reload_prompts(self):
        current_mtime = self._get_prompt_mtime()
        if current_mtime and current_mtime != self._prompts_mtime:
            try:
                self.prompts = self._load_prompts()
                self._prompts_mtime = current_mtime
                print("[LLM] 🔄 Prompt config reloaded")
            except Exception as exc:
                print(f"[LLM] ⚠️ Failed to reload prompts: {exc}")

    # --------------------------------------------------------------
    # LLM 호출 (task 단위)
    # --------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        task: str = "general",
        max_new_tokens: int = 512,
        system_override: str | None = None,
    ) -> str:
        self._maybe_reload_prompts()
        """YAML에 정의된 system prompt를 context별로 적용"""
        system_prompt = system_override or self.prompts.get(task, {}).get("system", "")
        if not system_prompt:
            print(f"[LLM] ⚠️ No system prompt found for task: {task}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        with self._lock:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            gen_ids = outputs[0][len(inputs.input_ids[0]):]
            result = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # cleanup
        if "</think>" in result:
            result = result.split("</think>")[-1]
        result = result.replace("\x00", "").replace("\u0000", "").strip()

        return result
