from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "google/gemma-2b-it"

print("Carregando modelo Gemma 2B...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype="float32",
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.3
)

def agent(prompt):
    result = generator(prompt)[0]["generated_text"]
    return result

if __name__ == "__main__":
    print("Digite sua pergunta:")
    user_input = input(">>> ")

    print("\nResposta do agente:\n")
    print(agent(user_input))
