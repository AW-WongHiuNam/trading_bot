"""Simple multi-agent conversation demo using ollama Python API.
Each agent has its own `system` role prompt and a private memory list.
Agents take turns calling `ollama.chat` (or `ollama.generate` if preferred).
"""
import time
import ollama

class Agent:
    def __init__(self, name: str, system_prompt: str, model: str = 'qwen2.5:14b'):
        self.name = name
        self.system = system_prompt
        self.private_memory = []  # list[str]
        self.model = model

    def make_messages(self, transcript):
        # transcript: list of (speaker, text)
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        # include private memory as additional system entries
        for mem in self.private_memory:
            msgs.append({"role": "system", "content": mem})
        # convert transcript: if speaker == self -> assistant else user
        for speaker, text in transcript:
            role = 'assistant' if speaker == self.name else 'user'
            content = f"{speaker}: {text}"
            msgs.append({"role": role, "content": content})
        return msgs


def extract_response_text(resp):
    # Try common shapes: attribute `response`, mapping-like items, or iterator of pairs
    # 1) attribute
    if hasattr(resp, 'response'):
        return getattr(resp, 'response')
    # 2) try dict-like
    try:
        if isinstance(resp, dict) and 'response' in resp:
            return resp['response']
    except Exception:
        pass
    # 3) iterate and look for tuple pairs
    try:
        parts = []
        for part in resp:
            # dataclass-like iteration may yield (key, value)
            try:
                if isinstance(part, tuple) and len(part) == 2 and part[0] == 'response':
                    return part[1]
                parts.append(str(part))
            except Exception:
                parts.append(str(part))
        if parts:
            return '\n'.join(parts)
    except TypeError:
        pass
    return str(resp)


def run_conversation(agent_a: Agent, agent_b: Agent, turns: int = 4, pause: float = 0.5):
    transcript = []  # list of (speaker, text)

    agents = [agent_a, agent_b]
    for i in range(turns):
        speaker = agents[i % 2]
        # prepare messages for this speaker
        messages = speaker.make_messages(transcript)
        print(f"\n--- {speaker.name} generating (turn {i+1}) ---")
        try:
            # use chat API
            resp = ollama.chat(model=speaker.model, messages=messages)
            text = extract_response_text(resp)
        except Exception as e:
            print('chat failed:', e)
            # fallback to generate with a prompt composed from last utterance
            last_text = transcript[-1][1] if transcript else 'Start the discussion.'
            prompt = f"{speaker.name}, reply to: {last_text}"
            try:
                resp = ollama.generate(model=speaker.model, prompt=prompt)
                text = extract_response_text(resp)
            except Exception as e2:
                print('generate fallback failed:', e2)
                text = f"(error generating reply: {e2})"

        # trim/clean text
        text = text.strip() if isinstance(text, str) else str(text)
        print(f"{speaker.name}: {text}\n")

        # append to transcript
        transcript.append((speaker.name, text))

        # store memories differently per agent example: each agent remembers their own lines and other's
        speaker.private_memory.append(f"I said: {text}")
        # the other agent may store a summarized memory (example)
        other = agents[(i + 1) % 2]
        other.private_memory.append(f"I heard {speaker.name} say: {text}")

        time.sleep(pause)

    return transcript


if __name__ == '__main__':
    a = Agent('Alice', system_prompt='You are Alice, a concise technical analyst.')
    b = Agent('Bob', system_prompt='You are Bob, a curious and friendly product manager.')
    conv = run_conversation(a, b, turns=6, pause=0.3)
    print('\n--- Final transcript ---')
    for s, t in conv:
        print(f"{s}: {t}")
