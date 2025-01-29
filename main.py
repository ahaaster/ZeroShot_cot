import dspy

LOCAL_MODELS = ["llama3.2:1b", "deepseek-r1:1.5b"]


def main():
    chosen_model = LOCAL_MODELS[0]
    lm = dspy.LM(
        f"ollama_chat/{chosen_model}", api_base="http://localhost:11434", api_key=""
    )

    x = lm("Hello World!", tempature=0.7)
    y = lm(messages=[{"role": "user", "content": "Hello World!"}])

    print(x)
    print()
    print(y)


if __name__ == "__main__":
    main()
