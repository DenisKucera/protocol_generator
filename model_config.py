import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """
    Spravuje API klíče, dynamicky ověřuje dostupné modely přímo u providerů
    a přiřazuje konkrétní modely jednotlivým agentům (vision, writer, analyst, inspector).
    """

    def __init__(self, env_path: Path = None):
        if env_path and env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv(override=True)

        gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
        self.keys = {
            "groq": os.getenv("GROQ_API_KEY", "").strip(),
            "gemini": gemini_key,
            "openai": os.getenv("OPENAI_API_KEY", "").strip(),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", "").strip(),
            # You can add another provider here...
        }
        # Odfiltrujeme pouze providery, kteří mají v .env vyplněný klíč
        self.active_providers = [p for p, k in self.keys.items() if k]

        # Slovník živých modelů načtených přímo z API jednotlivých providerů
        self.available_models: dict[str, list[str]] = {}

        # 1. Kontrola přítomnosti klíčů (nebo fallback na lokální Ollamu / sys.exit)
        self._verify_environment()

        # 2. Načtení aktuálně existujících modelů ze serverů (ochrana proti deprecation)
        self._fetch_all_active_models()

        # 3. Přiřazení modelů pro jednotlivé role agentů (podpora multi-provider režimu)
        self.vision_model = self._resolve_role("VISION_MODEL", role_type="vision")
        self.writer_model = self._resolve_role("WRITER_MODEL", role_type="text")
        self.analyst_model = self._resolve_role("ANALYST_MODEL", role_type="text")
        self.inspector_model = self._resolve_role("INSPECTOR_MODEL", role_type="text")
        self.planner_model   = self._resolve_role("PLANNER_MODEL", role_type="vision")

        self._print_summary()

    def _verify_environment(self) -> None:
        """Pokud chybí API klíče, zkusí lokální Ollamu, jinak vypíše chybu a ukončí běh."""
        if not self.active_providers:
            if self._check_local_ollama():
                print(" [ModelConfig] API klíč nebyl nalezen. Přepínám na lokální free API (Ollama).")
                self.active_providers = ["ollama"]
                return

            print("\n" + "=" * 65)
            print(" KRITICKÁ CHYBA: V souboru '.env' nebyl nalezen žádný API klíč!")
            print("Pro spuštění generátoru vytvořte soubor '.env' a vložte alespoň jeden API klíč:")
            print(" GROQ_API_KEY      ->  (https://console.groq.com)")
            print(" GEMINI_API_KEY    ->  (https://aistudio.google.com)")
            print(" OPENAI_API_KEY    ->  (https://platform.openai.com)")
            print(" ANTHROPIC_API_KEY ->  (https://console.anthropic.com)")
            print("=" * 65 + "\n")
            sys.exit(1)

    def _check_local_ollama(self) -> bool:
        """Ověří, zda na PC běží lokální free server Ollama (port 11434)."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _fetch_provider_models(self, provider: str) -> list[str]:
        """Pomocí match/case se dotáže API daného providera na aktuálně běžící modely."""
        try:
            match provider:
                case "groq":
                    resp = requests.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {self.keys['groq']}"},
                        timeout=5,
                    )
                    resp.raise_for_status()
                    return [f"groq/{m['id']}" for m in resp.json().get("data", [])]

                case "gemini":
                    resp = requests.get(
                        f"https://generativelanguage.googleapis.com/v1beta/models?key={self.keys['gemini']}",
                        timeout=5,
                    )
                    resp.raise_for_status()
                    # Vyfiltrujeme pouze modely podporující generování obsahu
                    return [
                        m["name"].replace("models/", "gemini/")
                        for m in resp.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]

                case "openai":
                    resp = requests.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {self.keys['openai']}"},
                        timeout=5,
                    )
                    resp.raise_for_status()
                    return [m["id"] for m in resp.json().get("data", [])]

                case "ollama":
                    resp = requests.get("http://localhost:11434/api/tags", timeout=3)
                    return [f"ollama/{m['name']}" for m in resp.json().get("models", [])]

                case "anthropic":
                    resp = requests.get(
                        "https://api.anthropic.com/v1/models",
                        headers={
                            "x-api-key": self.keys["anthropic"],
                            "anthropic-version": "2023-06-01",
                        },
                        timeout=5,
                    )
                    resp.raise_for_status()
                    return [f"anthropic/{m['id']}" for m in resp.json().get("data", [])]

                case _:
                    return []

        except Exception as e:
            print(f"  [ModelConfig] Nepodařilo se načíst aktualni seznam modelů pro '{provider}': {e}")
            return []

    def _fetch_all_active_models(self) -> None:
        for provider in self.active_providers:
            models = self._fetch_provider_models(provider)
            if models:
                self.available_models[provider] = models

    def _select_best_live_model(self, provider: str, role_type: str) -> str:
        """
        Prohledá živý seznam modelů od providera podle klíčových slov kvality.
        Pokud provider starý model vypne a nasadí novou verzi (např. llama-3.4 nebo gemini-2.5),
        tato funkce ho automaticky najde podle kmenového názvu!
        """
        live_list = self.available_models.get(provider, [])

        # Prioritní klíčová slova od nejlepších modelů po záložní
        match (provider, role_type):
            case ("gemini", _):
                # Gemini modely (Flash i Pro) jsou nativně multimodální (zvládají text i vision)
                priorities = ["2.5-flash", "2.0-flash", "1.5-pro", "1.5-flash", "flash"]
                fallback = "gemini/gemini-2.0-flash"
            case ("groq", "vision"):
                priorities = ["llama-4-scout", "llama-4-maverick", "vision", "90b-vision", "11b-vision"]
                fallback = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
            case ("groq", "text"):
                priorities = ["gpt-oss-120b", "llama-3.3-70b", "qwen-2.5-coder-32b", "70b", "8b-instant"]
                fallback = "groq/llama-3.3-70b-versatile"
            case ("openai", _):
                priorities = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
                fallback = "gpt-4o-mini"
            case ("anthropic", _):
                priorities = ["claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-haiku"]
                fallback = "anthropic/claude-3-5-sonnet-20241022"
            case _:
                priorities = []
                fallback = live_list[0] if live_list else "ollama/llama3"

        # Hledáme první model ze živého API, který obsahuje prioritní klíčové slovo
        for keyword in priorities:
            for live_model in live_list:
                if keyword in live_model.lower() and "whisper" not in live_model.lower() and "guard" not in live_model.lower():
                    return live_model

        # Pokud se neshoduje žádné klíčové slovo, vezmeme první textový model ze živého API nebo fallback
        return live_list[0] if live_list else fallback

    def _resolve_role(self, env_var_name: str, role_type: str) -> str:
        """
        Určí model pro konkrétního agenta:
        1. Pokud je v .env zadán konkrétní model nebo jen jméno providera (např. VISION_MODEL=gemini), použije ho.
        2. Pokud má uživatel více klíčů (např. Gemini i Groq), inteligentně rozloží práci!
        """
        user_choice = os.getenv(env_var_name, "").strip()

        # A) Uživatel v .env napsal jen název providera, např. WRITER_MODEL=groq nebo VISION_MODEL=gemini
        if user_choice.lower() in self.active_providers:
            return self._select_best_live_model(user_choice.lower(), role_type)

        # B) Uživatel v .env napsal konkrétní model (např. groq/llama-3.3-70b-versatile)
        if "/" in user_choice:
            # split("/", 1) rozdělí string POUZE podle prvního lomítka!
            # Z 'groq/openai/gpt-oss-120b' udělá: provider_prefix='groq', zbytek='openai/gpt-oss-120b'
            provider_prefix = user_choice.split("/", 1)[0].lower()
            if provider_prefix in self.active_providers:
                return user_choice
            print(f" [ModelConfig] Varování: Pro model '{user_choice}' chybí v .env odpovídající API klíč!")
            return user_choice

        # C) Automatický Multi-Provider Routing (když má uživatel v .env více klíčů najednou):
        # Např. Gemini je silné na Vision a dlouhý kontext, Groq je bleskový na psaní a inspekci
        if role_type == "vision" and "gemini" in self.active_providers:
            return self._select_best_live_model("gemini", "vision")
        if role_type == "text" and "groq" in self.active_providers:
            return self._select_best_live_model("groq", "text")

        # Jinak vezmeme prvního aktivního providera
        primary_provider = self.active_providers[0]
        return self._select_best_live_model(primary_provider, role_type)

    def _print_summary(self) -> None:
        print("\n [ModelConfig] konfigurace agentů...")
        print(f"   • Vision Curator : {self.vision_model}")
        print(f"   • Data Analyst   : {self.analyst_model}")
        print(f"   • Quarto Writer  : {self.writer_model}")
        print(f"   • Inspector      : {self.inspector_model}")
        print(f"   • Planner        : {self.planner_model}\n")