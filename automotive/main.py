import requests
import concurrent.futures
from rich.console import Console
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Agent:
    model: str = "gemma-4-e4b"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = field(default="NO_API_KEY", repr=False)
    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip('/') # we need that to avoid // when we connect it with /v1/chat/completions
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
    
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        r = requests.post(
            url, 
            headers=headers,
            json={"model": self.model, "messages": self.messages},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices")

        if not choices:
            raise RuntimeError("Model response missing choices")
        
        message = choices[0].get("message")
        if message is None:
            raise RuntimeError("Model response missing message")

        response = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": response})
        return response

class AutomotiveAssistant:
    def __init__(self) -> None:
        # Використовуємо три екземпляри LLM для отримання різних поглядів на ситуацію
        self.llm1 = Agent(
            system_prompt="Ти — перша LLM (Експерт з автомобільної безпеки). Надай технічні поради щодо дій водія у заданій ситуації. Відповідай коротко українською мовою."
        )
        self.llm2 = Agent(
            system_prompt="Ти — друга LLM (Досвідчений інструктор). Надай практичні поради для водія у заданій ситуації. Відповідай коротко українською мовою."
        )
        self.llm3 = Agent(
            system_prompt="Ти — третя LLM (Фахівець з надзвичайних ситуацій). Надай критично важливі вказівки для водія у заданій ситуації. Відповідай коротко українською мовою."
        )
        self.summarizer = Agent(
            system_prompt="Ти — головний помічник концепт-кара. Твоє завдання: узагальнити результати трьох обраних LLM щодо дій водія в певній ситуації, та сформувати єдину, чітку та зрозумілу інструкцію. Відповідай українською мовою."
        )
    
    def get_advice(self, situation: str) -> str:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.llm1.chat, situation)
            future2 = executor.submit(self.llm2.chat, situation)
            future3 = executor.submit(self.llm3.chat, situation)
            
            resp1 = future1.result()
            resp2 = future2.result()
            resp3 = future3.result()
            
        summary_prompt = f"""Ситуація водія: {situation}

Результат LLM 1 (Безпека):
{resp1}

Результат LLM 2 (Практика):
{resp2}

Результат LLM 3 (Екстрені дії):
{resp3}

Узагальни ці три відповіді та надай фінальну інструкцію для водія."""
        
        final_response = self.summarizer.chat(summary_prompt)
        return final_response

def main() -> None:
    assistant = AutomotiveAssistant()
    console = Console()

    console.print("[bold cyan]Бот-помічник концепт-кара (Automotive Concept Car)[/bold cyan]")
    console.print("Можу надати інструкції щодо дій водія при:")
    console.print(" 1. Потраплянні автівок в «мертву зону» дзеркал")
    console.print(" 2. Проблемах із тиском шин")
    console.print(" 3. Проблемах з очищенням лобового скла")
    console.print(" 4. Діях у випадку аварії")
    console.print(" 5. Перегріві двигуна [dim](Додатково)[/dim]")
    console.print(" 6. Відмові гальм [dim](Додатково)[/dim]")
    console.print(" 7. Зледенінні дороги [dim](Додатково)[/dim]")
    console.print("\nВведіть вашу ситуацію або номер, або 'quit' для виходу.")

    situations_map = {
        "1": "Потрапляння автівок в «мертву зону» дзеркал",
        "2": "Проблеми із тиском шин",
        "3": "Проблеми з очищенням лобового скла",
        "4": "Дії у випадку аварії",
        "5": "Перегрів двигуна",
        "6": "Відмова гальм",
        "7": "Зледеніння дороги"
    }

    while True:
        console.print("\n[green]Водій:[/green] ", end="")
        user_input = console.input()
        
        if user_input.strip().lower() in {"quit", "exit", "вихід"}:
            console.print("[dim]Щасливої дороги![/dim]")
            break
            
        if user_input.strip() in situations_map:
            user_input = situations_map[user_input.strip()]
            console.print(f"[dim]Обрана ситуація: {user_input}[/dim]")
        
        with console.status("[dim]Узагальнюю результати трьох LLM...[/dim]", spinner="arc"):
            try:
                response = assistant.get_advice(user_input).strip()
            except Exception as e:
                response = f"Помилка при генерації відповіді: {e}"

        console.print(f"\n[bold blue]Помічник:[/bold blue]\n{response}")

if __name__ == "__main__":
    main()
