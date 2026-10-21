import os
import subprocess
from pathlib import Path
from autogen import ConversableAgent

class BaseLabAgent:
    """Base class pro všechny laboratorní AI agenty."""

    def __init__(self, name: str, system_message: str, config_list: list, max_tokens: int = 4096):
        self.name = name
        self.agent = ConversableAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "config_list": config_list,
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
        )

    def ask(self, prompt: str) -> str:
        """Odešle dotaz agentovi a vrátí očištěný text."""
        reply = self.agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        return self.clean_fences(content.strip())

    @staticmethod
    def clean_fences(text: str) -> str:
        """Odstraní obalovací markdown bloky (```python, ```markdown apod.), pokud obalují celý výstup."""
        text = text.strip()
        for prefix in ["```python", "```markdown", "```quarto", "```qmd", "```"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                return text
        return text


class DataAnalystAgent(BaseLabAgent):
    """Fáze 1: Píše a spouští čistý Python skript pro zpracování dat a export grafů."""

    def __init__(self, config_list: list, valid_files: list[str], task_rules: list[str]):
        rules_str = "\n".join([f"- {r}" for r in task_rules])
        sys_msg = f"""Jsi Python Data Scientist pro laboratorní měření na VUT FEKT.
        Tvým jediným úkolem je napsat samostatný, spustitelný Python skript (čistý kód bez Markdown textu okolo).
        Skript musí:
        1. Načíst data ze souboru {valid_files}.
        2. Provést výpočty podle pravidel úlohy a uložit souhrnnou tabulku výsledků (zaokrouhlenou na 3 des. místa) jako JSON seznam slovníků do 'processed/summary.json' (orient='records', force_ascii=False, indent=2).
        3. Pro každé měření vykreslit graf pomocí Matplotlibu (`plt.style.use('seaborn-v0_8-whitegrid')`) a uložit ho jako PDF do 'processed/graf_<nazev>.pdf' pomocí `plt.savefig(..., bbox_inches='tight')` a `plt.close()`.

        PRAVIDLA VÝPOČTU A GRAFŮ PRO TUTO ÚLOHU:
        {rules_str}"""
        super().__init__(name="Data_Analyst", system_message=sys_msg, config_list=config_list, max_tokens=4096)

    def execute(self, zadani_text: str, data_info: str, max_retries: int = 3) -> bool:
        prompt = (
            f"ZADÁNÍ:\n{zadani_text}\n\n"
            f"STRUKTURA DAT:\n{data_info}\n\n"
            f"Napiš čistý Python kód skriptu process_data.py."
        )

        for attempt in range(1, max_retries + 1):
            code = self.ask(prompt)
            with open("process_data.py", "w", encoding="utf-8") as f:
                f.write(code)

            # Samostatné spuštění a otestování skriptu
            res = subprocess.run(["python", "process_data.py"], capture_output=True, text=True, encoding="utf-8")
            if res.returncode == 0 and os.path.exists("processed/summary.json"):
                print(f"  ✅ [Data_Analyst] Skript process_data.py proběhl úspěšně (pokus {attempt}).")
                return True

            print(f"  ⚠️ [Data_Analyst] Chyba při běhu skriptu (pokus {attempt}/{max_retries}):\n{res.stderr}")
            prompt = (
                f"Tvůj skript spadl s touto chybou:\n{res.stderr}\n\n"
                f"Zde je tvůj předchozí kód:\n{code}\n\n"
                f"Oprav ho a vrať kompletní funkční Python skript (bez ```python bloků)."
            )

        raise RuntimeError("❌ DataAnalystAgent nedokázal vygenerovat funkční výpočetní skript ani po 3 pokusech.")


class QuartoWriterAgent(BaseLabAgent):
    """Fáze 2: Sestavuje první návrh protokolu (.qmd) z již vypočtených dat a hotových grafů."""

    def __init__(self, config_list: list, global_rules: list[str], task_rules: list[str]):
        all_rules = "\n".join([f"- {r}" for r in (global_rules + task_rules)])
        sys_msg = f"""Jsi odborný autor laboratorních protokolů v systému Quarto (.qmd) na VUT FEKT.
        Veškeré výpočty a grafy už za tebe připravil datový analytik!
        Tvým úkolem je napsat tělo souboru protokol.qmd (sekce '# Teoretický úvod', '# Metodika', '# Výsledky', '# Diskuze', '# Závěr').

        JAK VLOŽIT HOTOVÁ DATA A GRAFY:
        - V sekci '# Výsledky' načti 'processed/summary.json' přes `df = pd.read_json('processed/summary.json')` a vypiš tabulku pomocí `display(Markdown(df.to_markdown(index=False)))`.
        - Grafy už NEPOČÍTEJ, pouze je vlož ze seznamu vygenerovaných souborů přes Markdown: `![V-A charakteristika](processed/graf_nazev.pdf){{width=80%}}`.
        - V sekcích '# Diskuze' a '# Závěr' komentuj PŘESNÁ ČÍSLA z dodaného JSON souhrnu!
        - Pokud si v jakékoliv části nejsi jistý (chybí informace v zadání), vlož značku: `[NEJISTOTA: stručný popis, co chybí]`.

        STRIKTNÍ PRAVIDLA FORMÁTOVÁNÍ:
        {all_rules}"""
        super().__init__(name="Quarto_Writer", system_message=sys_msg, config_list=config_list, max_tokens=8192)

    def execute(self, zadani_text: str, computed_summary: str, generated_plots: list[str], output_path: Path) -> str:
        prompt = (
            f"ZADÁNÍ ÚLOHY:\n{zadani_text}\n\n"
            f"HOTOVÉ VÝSLEDKY VÝPOČTŮ (processed/summary.json):\n{computed_summary}\n\n"
            f"VYGENEROVANÉ GRAFY K VLOŽENÍ:\n{generated_plots}\n\n"
            f"Sestav kompletní návrh protokolu od '# Teoretický úvod' až po '# Závěr'."
        )
        draft_content = self.ask(prompt)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(draft_content)
        print(f"  ✅ [Quarto_Writer] Návrh uložen do '{output_path.name}'.")
        return draft_content


class ProtocolInspectorAgent(BaseLabAgent):
    """Fáze 3: Opravuje chyby ze Scoreboardu a řeší značky [NEJISTOTA: ...] v návrhu."""

    def __init__(self, config_list: list, global_rules: list[str]):
        rules_str = "\n".join([f"- {r}" for r in global_rules])
        sys_msg = f"""Jsi přísný kontrolor laboratorních protokolů VUT FEKT (Protocol Inspector).
        Dostaneš návrh protokolu (.qmd), seznam syntaktických chyb ze Scoreboardu a případné značky `[NEJISTOTA: ...]` od autora.
        Tvůj úkol:
        1. Oprav všechny nahlášené chyby ze Scoreboardu a dodrž přitom pravidla:
        {rules_str}
        2. Zkontroluj každou značku `[NEJISTOTA: ...]`. Pokud odpověď najdeš v zadání nebo ve vypočtených datech, nahraď značku přesným faktem.
        3. Pokud informaci nelze z podkladů zjistit, nahraď ji výrazným upozorněním: `> **⚠️ K RUČNÍMU DOPLNĚNÍ:** [popis, co má uživatel doplnit]`.
        4. Vrať kompletní, vyčištěný dokument od '# Teoretický úvod' až po '# Závěr'."""
        super().__init__(name="Protocol_Inspector", system_message=sys_msg, config_list=config_list, max_tokens=8192)

    def execute(self, draft_qmd: str, penalties: list[str], zadani_text: str, computed_summary: str) -> str:
        penalties_str = "\n".join(penalties) if penalties else "Žádné syntaktické chyby."
        prompt = (
            f"PENALIZACE ZE SCOREBOARDU K OPRAVĚ:\n{penalties_str}\n\n"
            f"ZADÁNÍ PRO OVĚŘENÍ FAKTŮ:\n{zadani_text}\n\n"
            f"VYPOČTENÁ DATA:\n{computed_summary}\n\n"
            f"NÁVRH PROTOKOLU K REVIZI:\n{draft_qmd}"
        )
        return self.ask(prompt)