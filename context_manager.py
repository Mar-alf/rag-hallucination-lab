FEW_SHOT_EXAMPLES = """
示例1：
用户问：白色马尾毛有什么特点？
回答：白色马尾毛透气性强，支撑力好，适合喜欢偏硬睡感的人。

示例2：
用户问：酷布7环和9环差在哪？
回答：酷布7环是七环设计，弹性适中；酷布9环支撑更强，适合腰椎问题人群。

示例3：
用户问：天然新疆棉适合谁？
回答：天然新疆棉柔软亲肤，适合敏感肌和喜欢软睡感的人。
"""


def build_prompt(question, retrieved_docs):
    context = "\n".join(retrieved_docs)
    return f"""{FEW_SHOT_EXAMPLES}

请严格按示例的风格回答。只能使用下面的材料说明，无依据时明确说"材料库中未提及"。

材料说明：
{context}

用户问：{question}
回答："""


def summarize_observation(observation, max_len=100):
    if len(observation) <= max_len:
        return observation
    return observation[:max_len] + "...(已压缩)"


class ConversationHistory:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []
        self.summary = ""

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.window_size:
            oldest = self.history.pop(0)
            self.summary += summarize_observation(oldest["content"]) + "；"

    def get_context(self):
        return {"summary": self.summary, "recent": self.history}
