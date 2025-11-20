from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "distilgpt2"

print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

def agent(prompt):
    response = generator(prompt)[0]["generated_text"]
    # Remove o prompt repetido
    response_clean = response[len(prompt):].strip()
    return response_clean

if __name__ == "__main__":
    print("Digite sua pergunta:")
    user_input = input(">>> ")

    if not user_input.strip():
        print("Por favor, digite uma pergunta.")
        exit()

    print("\nResposta do agente:\n")
    print(agent(user_input))
