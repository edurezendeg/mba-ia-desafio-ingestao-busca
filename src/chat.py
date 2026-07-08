"""CLI de perguntas e respostas sobre o conteudo do PDF.

Execucao (a partir da raiz do projeto):
    python src/chat.py

Digite 'sair' (ou Ctrl+C / Ctrl+D) para encerrar.
"""

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main() -> None:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("=" * 60)
    print(" Chat sobre o PDF (responde somente com base no documento).")
    print(" Digite 'sair' para encerrar.")
    print("=" * 60)

    while True:
        print("\nFaça sua pergunta:\n")
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Até logo!")
            break

        try:
            answer = chain.invoke(question)
        except Exception as error:  # noqa: BLE001 - feedback amigavel no CLI
            print(f"RESPOSTA: [erro ao consultar] {error}")
            continue

        print(f"RESPOSTA: {answer}")
        print("\n---")


if __name__ == "__main__":
    main()
