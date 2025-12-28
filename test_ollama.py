import sys
import traceback

try:
    import ollama
    from ollama import Client
except Exception as e:
    print('Failed to import ollama Python package:', e)
    raise

def main():
    print('\nUsing ollama Python package')
    try:
        client = Client()
        print('Client created:', type(client))
    except Exception as e:
        print('Failed to create Client:', e)
        client = None

    model = 'qwen2.5:14b'
    prompt = 'Say hello and include the model name.'

    # Try `generate` (accepts `prompt`) first
    try:
        resp = ollama.generate(model=model, prompt=prompt)
        print('\nollama.generate response (raw):')
        print(resp)
        if hasattr(resp, '__iter__') and not isinstance(resp, str):
            print('\nIterating generate response parts:')
            for part in resp:
                print(part)
        return
    except Exception as e:
        print('ollama.generate failed:', e)
        traceback.print_exc()

    # Fallback to chat (messages)
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = ollama.chat(model=model, messages=messages)
        print('\nollama.chat response (raw):')
        print(resp)
        if hasattr(resp, '__iter__') and not isinstance(resp, str):
            print('\nIterating chat response parts:')
            for part in resp:
                print(part)
    except Exception as e:
        print('ollama.chat failed:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()
