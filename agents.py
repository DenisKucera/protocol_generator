import os
import subprocess
import sys
import base64
import json
from pathlib import Path
from autogen import ConversableAgent
from PIL import Image, ImageOps
from openai import OpenAI

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

    KRITICKÁ PRAVIDLA PRO BEZCHYBNÉ NAČTENÍ EXCELU (ZABRAŇUJE TypeError a ValueError):
    1. Vstupní Excely mohou mít nadpisy na více řádcích, jednotky pod hlavičkou nebo více tabulek vedle sebe. Pečlivě si prohlédni dodanou ukázku surových buněk (`header=None`)!
    2. Při extrakci jakéhokoliv sloupce z DataFrame jej VŽDY nejprve převeď na čistá čísla (tím se automaticky zahodí textové řádky s jednotkami, nadpisy i prázdné buňky):
       `col_num = pd.to_numeric(df.iloc[:, col_idx].astype(str).str.replace(',', '.'), errors='coerce')`
       a spárované sloupce (např. U a I) očisti od NaN společně: `valid = (~U.isna()) & (~I.isna()); U = U[valid].to_numpy(); I = I[valid].to_numpy()`.
    3. NIKDY nepoužívej v kódu `raise ValueError(...)` ani `raise RuntimeError(...)` při hledání názvů sloupců! Pokud názvy sloupců neodpovídají, vyber sloupce přímo podle jejich číselného indexu (`df.iloc[:, idx]`) viditelného v surové ukázce dat!
    4. Pro výpočty a regresi používej výhradně `numpy` a `pandas` (např. `np.polyfit`), NIKDY neimportuj `scipy`.
    5. Ulož souhrnnou tabulku výsledků (zaokrouhlenou na 3 des. místa) jako JSON seznam slovníků do 'processed/summary.json' (`orient='records'`, `force_ascii=False`, `indent=2`).
    6. Vykresli grafy pomocí Matplotlibu (`plt.style.use('seaborn-v0_8-whitegrid')`) a ulož je jako PDF do 'processed/graf_<nazev>.pdf' pomocí `plt.savefig(..., bbox_inches='tight')` a `plt.close()`.

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

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Samostatné spuštění a otestování skriptu
            res = subprocess.run([sys.executable, "process_data.py"], 
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env)
            # Uvnitř DataAnalystAgent.execute() v agents.py:
            if res.returncode == 0 and os.path.exists("processed/summary.json"):
                with open("processed/summary.json", "r", encoding="utf-8") as sf:
                    summary_text = sf.read()
                
                # TVRDÁ KONTROLA KVALITY VÝPOČTŮ (GIGO prevence):
                if "nan" in summary_text.lower() or "null" in summary_text.lower() or len(summary_text) < 20:
                    print(f"  ⚠️ [Data_Analyst] Skript sice doběhl, ale summary.json obsahuje NaN/null hodnoty (pokus {attempt}/{max_retries})!")
                    prompt = (
                        f"Tvůj skript doběhl, ale vygenerovaný 'processed/summary.json' obsahuje neplatné hodnoty (NaN nebo null):\n"
                        f"{summary_text}\n\n"
                        f"Zde je tvůj kód:\n{code}\n\n"
                        f"PŘÍČINA: Pravděpodobně filtruješ data příliš úzkou podmínkou (např. špatný rozsah proudu/napětí) nebo špatně indexuješ sloupce v Excelu. "
                        f"Oprav výpočet tak, aby všechny číselné parametry vyšly jako reálná konečná čísla (žádné NaN/null)!"
                    )
                    continue

                print(f"  ✅ [Data_Analyst] Skript process_data.py proběhl úspěšně a data jsou validní (pokus {attempt}).")
                return True

            print(f"  ⚠️ [Data_Analyst] Chyba při běhu skriptu (pokus {attempt}/{max_retries}):\n{res.stderr}")
            prompt = (
                f"Tvůj skript spadl s touto chybou:\n{res.stderr}\n\n"
                f"Zde je tvůj předchozí kód:\n{code}\n\n"
                f"Oprav ho a vrať kompletní funkční Python skript (bez ```python bloků)."
            )

        raise RuntimeError("❌ DataAnalystAgent nedokázal vygenerovat funkční výpočetní skript ani po 3 pokusech.")


class QuartoWriterAgent(BaseLabAgent):
    """Fáze 2: Sestavuje návrh protokolu (.qmd) jako čistý Quarto Markdown z hotových dat."""

    def __init__(self, config_list: list, global_rules: list[str], task_rules: list[str]):
        all_rules = "\n".join([f"- {r}" for r in (global_rules + task_rules)])
        sys_msg = f"""Jsi odborný autor laboratorních protokolů v systému Quarto (.qmd) na VUT FEKT.
    Veškeré výpočty a grafy už za tebe připravil datový analytik!
    Tvým úkolem je napsat tělo souboru protokol.qmd (sekce '# Teoretický úvod', '# Metodika', '# Výsledky', '# Diskuze', '# Závěr').

    KRITICKÁ PRAVIDLA PRO STRUKTURU (.qmd):
    1. NIKDY na začátek dokumentu nepiš YAML hlavičku (žádné `---`, `title:`, `author:`)! Začni rovnou na prvním řádku nadpisem `# Teoretický úvod`.
    2. Do dokumentu už NEPIŠ ŽÁDNÉ spustitelné bloky ```` ```{{python}} ````! Všechna vypočtená čísla z `processed/summary.json` už vidíš v promptu – vypiš souhrnnou tabulku v sekci '# Výsledky' přímo jako čistou Markdown tabulku (`| Sloupec 1 | Sloupec 2 |...`) a konkrétní čísla z ní rovnou dosaď přímo do textu v sekcích '# Diskuze' a '# Závěr'!
    3. Grafy vlož v sekci '# Výsledky' přes Markdown: `![Popis grafu](processed/graf_nazev.pdf){{width=75%}}`. Mezi každými dvěma obrázky MUSÍ být vždy prázdný řádek (nový odstavec), jinak se v PDF překryjí vedle sebe!
    4. Značku `[NEJISTOTA: ...]` použij MAXIMÁLNĚ 1x v celém dokumentu, a to pouze tehdy, pokud v zadání vysloveně chybí klíčový parametr. Nikdy nepoužívej více značek v jedné větě!
    5. VKLÁDÁNÍ FOTOGRAFIÍ ZE SLOŽKY 'img/' (POKUD JSOU V KATALOGU OBRÁZKŮ):
    - Pokud dostaneš 'KATALOG FOTOGRAFIÍ Z LABORATOŘE', musíš do dokumentu vložit VŠECHNY fotografie z katalogu v jejich přesném pořadí!
    - Každou skupinu fotografií uveď podkapitolou (např. `## Příprava zařízení a materiálu`) a souvislým odborným odstavcem, ve kterém popíšeš prováděný krok a odkážeš se na číslo obrázku v závorce, např. `(Obr. 1)` nebo `viz Obr. 2`.
    - Samotný obrázek vlož VŽDY na samostatný odstavec (s prázdným řádkem nad i pod!) přesně v tomto formátu:
      `![Přesný text z pole caption bez prefixu Obr. X](cesta_z_pole_file){{width=62% fig-align="center"}}`
    - NIKDY nepiš dovnitř hranatých závorek `![Obr. 1. ...]` slovo 'Obr. 1.', protože LaTeX ho tam doplňuje automaticky (vzniklo by zdvojené 'Obr. 1.: Obr. 1.')!
    6. Pokud chceš v Python bloku vypsat delší text přes display(Markdown(...)), použij buď trojité uvozovky (rf\"\"\"...\"\"\"), nebo řádek nerozděluj Enterem uvnitř uvozovek – použij explicitní \n pro nový řádek.
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

class VisionCuratorAgent:
    """
    Fáze 1B: Zpracuje fotografie ve složce img/, opraví jejich EXIF rotaci,
    pomocí Vision modelu rozpozná, co na nich je, a sestaví 'processed/images_catalog.json'.
    """

    def __init__(self, api_key: str, vision_model: str = "qwen/qwen3.8-27b"):
        self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.vision_model = vision_model

    @staticmethod
    def _prepare_and_encode_image(img_path: str) -> str:
        """Opraví EXIF otočení fotky z telefonu, zmenší rozlišení pro PDF/API a vrátí base64."""
        with Image.open(img_path) as img:
            # 1. Oprava otočení z mobilu (EXIF Orientation tag)
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # 2. Zmenšení příliš velkých fotek (max 1400 px), aby PDF nemělo stovky MB
            img.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            img.save(img_path, format="JPEG", quality=85)

        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def execute(self, image_files: list[str], zadani_text: str, task_rules: list[str]) -> list[dict]:
        catalog = []
        rules_str = "\n".join([f"- {r}" for r in task_rules])

        print(f"  📸 [Vision_Curator] Nalezeno {len(image_files)} obrázků ve složce 'img/'. Analyzuji obsah...")

        for idx, img_path in enumerate(image_files, start=1):
            b64_img = self._prepare_and_encode_image(img_path)
            filename = os.path.basename(img_path)

            prompt = f"""Jsi odborný asistent v laboratořích VUT FEKT.
        Analyzuješ fotografii č. {idx} (soubor '{filename}') pro laboratorní protokol.

        KONTEXT ZE ZADÁNÍ ÚLOHY:
        {zadani_text[:2500]}

        SPECIFICKÁ PRAVIDLA ÚLOHY:
        {rules_str}

        Prohlédni si fotografii (všímej si přístrojů, displejů, hodnot, chemikálií a prováděné činnosti) a vrať ČISTÝ JSON objekt (bez ```json bloků) s těmito klíči:
        {{
          "obr_cislo": {idx},
          "file": "{img_path}",
          "sekce": "Název logické podkapitoly, kam fotka v postupu patří (např. Příprava zařízení a materiálu, Osvit síta, Optická kontrola apod.)",
          "caption": "Stručný, přesný odborný popisek pod obrázek (BEZ prefixu 'Obr. X.', ten doplní LaTeX sám! Např.: 'Detail displeje sušičky zobrazující průběh procesu sušení a teploty')",
          "detailni_popis": "Co přesně je na fotce vidět včetně případných přečtených čísel z displeje/mikroskopu (pro autora textu protokolu)."
        }}"""

            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_tokens=600,
                )
                raw_json = BaseLabAgent.clean_fences(response.choices[0].message.content.strip())
                item = json.loads(raw_json)
                item["obr_cislo"] = idx
                item["file"] = img_path
            except Exception as e:
                print(f"  ⚠️ [Vision_Curator] Záložní popisek pro {filename} ({e})")
                clean_name = os.path.splitext(filename)[0].replace("_", " ")
                item = {
                    "obr_cislo": idx,
                    "file": img_path,
                    "sekce": "Praktická část",
                    "caption": f"Fotodokumentace kroku měření ({clean_name})",
                    "detailni_popis": f"Fotografie {clean_name}",
                }

            catalog.append(item)
            print(f"    ↳ Obr. {idx} ({filename}): {item['caption']}")

        with open("processed/images_catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

        return catalog