# 🤖 LLMAgent - Projeto de Agente de Linguagem com Gemma

Este projeto implementa um Agente de Linguagem utilizando o modelo **Gemma-2b-it** da Google. O objetivo principal é carregar o modelo e criar um pipeline de geração de texto para tarefas de Inteligência Artificial.

## 🚀 Começando

Siga estas instruções para configurar e rodar o projeto em sua máquina local.

### Pré-requisitos

Você precisará ter o Python instalado. É altamente recomendado o uso de um ambiente virtual (`venv`).

1.  **Crie e Ative o Ambiente Virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    .\venv\Scripts\activate   # Windows
    ```

2.  **Instale as Dependências:**
    *(Assumindo que você usa `transformers`, `torch` e `huggingface_hub`)*
    ```bash
    pip install transformers torch huggingface-hub
    ```

### 🔑 Autenticação no Hugging Face

O modelo `google/gemma-2b-it` é um modelo de acesso restrito ("Gated Model"). Você deve estar autenticado no Hugging Face Hub para baixá-lo.

1.  **Obtenha o Token:** Certifique-se de que você **aceitou os termos do modelo** na página do Hugging Face e **gerou um Token de Acesso** (com permissão de **Read**).
2.  **Faça o Login:** Use a Command Line Interface (CLI) para autenticar seu ambiente:
    ```bash
    huggingface-cli login
    # Cole o seu Token de Acesso quando solicitado.
    ```
    *(Você também pode usar o novo comando: `hf auth login`)*

### ⚙️ Executando o Projeto

O script principal é o `agent_app.py`.

Para executar o agente e carregar o modelo:

```bash
python agent_app.py