import time
import json
import re
import argparse
from pathlib import Path
from openai import OpenAI

# =========================
# 配置
# =========================
INPUT_FILE = "experiment_data/candidate_functions.json"
OUTPUT_DIR = Path("generated_tests")
PROMPT_DIR = Path("prompts")

STRATEGIES = ["zero_shot", "few_shot", "role_based", "CoT"]

client = OpenAI()


# =========================
# 工具函数
# =========================
def load_functions():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template(strategy):
    path = PROMPT_DIR / f"{strategy}.txt"
    return path.read_text(encoding="utf-8")


def build_prompt(template, function_source):
    return template.replace("{{FUNCTION_SOURCE}}", function_source)

def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# =========================
# 提取测试代码（关键）
# =========================
def extract_tests(code_text):
    """
    从GPT输出中提取pytest测试代码
    """
    # 优先提取 ```python ``` 代码块
    code_blocks = re.findall(r"```python(.*?)```", code_text, re.DOTALL)

    if code_blocks:
        return code_blocks[0].strip()

    # fallback：直接返回全部内容
    return code_text.strip()


# =========================
# 主逻辑
# =========================
def generate_tests(strategy):
    assert strategy in STRATEGIES, f"Invalid strategy: {strategy}"

    functions = load_functions()
    template = load_prompt_template(strategy)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"generated_tests_{strategy}.json"

    results = []

    for idx, func in enumerate(functions):
        print(f"[{idx+1}/{len(functions)}] {func['name']} ({strategy})")

        prompt = build_prompt(template, func["source"])

        try:
            time.sleep(1)
            raw_output = call_gpt(prompt)
            tests_code = extract_tests(raw_output)

        except Exception as e:
            print(f"  ERROR: {e}")
            raw_output = ""
            tests_code = ""

        result = {
            "project": func["project"],
            "module": func["module"],
            "name": func["name"],
            "source": func["source"],
            "tests_source": tests_code,
            "raw_output": raw_output   # ⭐建议保留，方便debug/论文分析
        }

        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_file}")


# =========================
# CLI入口
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        required=True,
        choices=STRATEGIES,
        help="Prompt strategy"
    )

    args = parser.parse_args()

    generate_tests(args.strategy)

    # generate_tests("zero_shot")