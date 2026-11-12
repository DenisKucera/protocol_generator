import os
import subprocess
import sys
import re
import base64
import json
import easyocr
import litellm
import warnings
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
#from autogen import ConversableAgent
from PIL import Image, ImageOps
from io import BytesIO
#from openai import OpenAI
litellm.drop_params = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
class BaseLabAgent:
    """Base class pro všechny LLM agenty."""

    def __init__(self, model_name: str, system_prompt: str = "", temperature: float = 0.2):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def call_model(
        self,
        user_prompt: str,
        system_prompt: str = None,
        json_mode: bool = False,
        temperature: float = None,
    ) -> str:
        """
        Univerzální metoda pro odeslání dotazu přes LiteLLM.
        """
        sys_p = system_prompt if system_prompt is not None else self.system_prompt
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
            messages.append({"role": "user", "content": user_prompt})

        # response_format vynucujeme jen pro OpenAI a Groq, ostatní se spoléhají na prompt
        use_json_fmt = json_mode and any(p in self.model_name.lower() for p in ["groq", "openai","gemini"])

        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=8192,
                response_format={"type": "json_object"} if use_json_fmt else None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" [{self.__class__.__name__}] Chyba při volání modelu {self.model_name}: {e}")
            raise e

    @staticmethod
    def strip_code_fences(text: str, tag: str = "") -> str:
        """
        Odstraní obalující bloky ```tag ... ``` z odpovědi LLM.
        """
        pattern = rf"^```(?:{tag})?\s*\n(.*?)\n```$"
        match = re.search(pattern, text.strip(), flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Pokud model neuzavřel blok nebo je na začátku text navíc
        text = re.sub(rf"^```(?:{tag})?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        return text

class TaskPlannerAgent(BaseLabAgent):
    """
    Přečte surový text zadání (zadani.pdf), oddělí teoretickou omáčku
    a vyextrahuje striktní JSON checklist praktických úkolů.
    """
    def __init__(self, model_name: str, task_rules: list[str] = None, forbidden_words: list[str] = None):
        rules_str = "\n".join([f"- {r}" for r in (task_rules or [])])
        forbidden_str = ", ".join([f"'{w}'" for w in forbidden_words]) if forbidden_words else "žádná"
        system_prompt = f"""Jsi analytik laboratorních úloh na VUT FEKT.
            Tvým úkolem je přečíst zadání a oddělit teoretický úvod (včetně alternativních technologií, které se v laboratoři reálně nedělají) od skutečných praktických kroků.
            Respektuj tato pravidla úlohy:{rules_str} 
            a nikdy nepoužívej tato slova:{forbidden_str}"""
        super().__init__(model_name=model_name, system_prompt=system_prompt, temperature=0.1)
        #self.forbidden_words = forbidden_words or []

    def execute(self, zadani_text: str, output_plan_path: Path) -> dict:
            prompt = f"""ZADÁNÍ ÚLOHY:
            {zadani_text[:4500]}

            Vrať ČISTÝ JSON objekt v této struktuře:
            {{
            "teorie_shrnuti": "3 až 4 věty shrnující čistě teoretický základ úlohy pro kapitolu Teoretický úvod.",
            "pozadovane_kroky": [
                {{
                "id": 1,
                "nazev_sekce": "Název technologického kroku (např. Dělení substrátů)",
                "popis_ukolu": "Co přesně má student v tomto kroku udělat/změřit",
                "klicova_slova": ["rýhování", "diamant", "substrát"]
                }}
            ]
            }}"""
            raw_res = self.call_model(prompt, json_mode=True)
            clean_res = self.strip_code_fences(raw_res, tag="json")
            plan_data = json.loads(clean_res)

            output_plan_path.parent.mkdir(parents=True, exist_ok=True)
            output_plan_path.write_text(json.dumps(plan_data, ensure_ascii=False, indent=2), encoding="utf-8")
            return plan_data
    
class DataAnalystAgent(BaseLabAgent):
    """Fáze 1A: Píše a spouští čistý Python skript pro zpracování dat a export grafů."""

    def __init__(self, model_name: str, valid_files: list[str], task_rules: list[str]):
        rules_str = "\n".join([f"- {r}" for r in (task_rules or [])])
        files_str = ", ".join([f"'{f}'" for f in (valid_files or [])])

        system_prompt = f"""Jsi Python Data Scientist pro laboratorní měření na VUT FEKT.
        Tvým jediným úkolem je napsat samostatný, spustitelný Python skript (čistý kód bez Markdown textu okolo).

        DOSTUPNÉ DATOVÉ SOUBORY VE SLOŽCE data/ (pracuj VÝHRADNĚ s těmito soubory):
        [{files_str}]

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
        super().__init__(model_name=model_name, system_prompt=system_prompt, temperature=0.1)

    def execute(self, plan_or_zadani: str | dict, data_info: str, max_retries: int = 3) -> bool:
        # Pokud posíláme výstup z TaskPlannerAgenta (dict), převedeme ho na přehledný JSON text
        if isinstance(plan_or_zadani, dict):
            zadani_context = json.dumps(plan_or_zadani.get("pozadovane_kroky", []), ensure_ascii=False, indent=2)
        else:
            zadani_context = str(plan_or_zadani)

        prompt = (
            f"POŽADOVANÉ KROKY A VÝPOČTY ZE ZADÁNÍ:\n{zadani_context}\n\n"
            f"STRUKTURA NAMĚŘENÝCH DAT:\n{data_info}\n\n"
            f"Napiš čistý Python kód skriptu process_data.py."
        )

        os.makedirs("processed", exist_ok=True)

        for attempt in range(1, max_retries + 1):
            # 1. Zavoláme LLM přes novou metodu z BaseLabAgent a očistíme kód od ```python bloků
            raw_response = self.call_model(prompt)
            code = self.strip_code_fences(raw_response, tag="python")

            with open("process_data.py", "w", encoding="utf-8") as f:
                f.write(code)

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # 2. Samostatné spuštění a otestování vygenerovaného skriptu
            res = subprocess.run(
                [sys.executable, "process_data.py"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            # 3. Kontrola úspěšného běhu a GIGO validace výstupního JSONu
            if res.returncode == 0 and os.path.exists("processed/summary.json"):
                with open("processed/summary.json", "r", encoding="utf-8") as sf:
                    summary_text = sf.read()

                if "nan" in summary_text.lower() or "null" in summary_text.lower() or len(summary_text) < 20:
                    print(f" [Data_Analyst] Skript doběhl, ale summary.json obsahuje NaN/null (pokus {attempt}/{max_retries})!")
                    prompt = (
                        f"Tvůj skript doběhl, ale vygenerovaný 'processed/summary.json' obsahuje neplatné hodnoty (NaN nebo null):\n"
                        f"{summary_text}\n\n"
                        f"Zde je tvůj kód:\n{code}\n\n"
                        f"PŘÍČINA: Pravděpodobně filtruješ data příliš úzkou podmínkou nebo špatně indexuješ sloupce v Excelu. "
                        f"Oprav výpočet tak, aby všechny číselné parametry vyšly jako reálná konečná čísla (žádné NaN/null)!"
                    )
                    continue

                print(f" [Data_Analyst] Skript process_data.py proběhl úspěšně a data jsou validní (pokus {attempt}).")
                return True

            # 4. Pokud skript spadl na chybu v Pythonu, předáme chybovou hlášku (stderr) zpět agentovi k opravě
            print(f" [Data_Analyst] Chyba při běhu skriptu (pokus {attempt}/{max_retries}):\n{res.stderr}")
            prompt = (
                f"Tvůj skript spadl s touto chybou:\n{res.stderr}\n\n"
                f"Zde je tvůj předchozí kód:\n{code}\n\n"
                f"Oprav ho a vrať kompletní funkční Python skript (bez ```python bloků)."
            )

        raise RuntimeError(" DataAnalystAgent nedokázal vygenerovat funkční výpočetní skript ani po 3 pokusech.")


class QuartoWriterAgent(BaseLabAgent):
    """Fáze 2: Sestavuje návrh protokolu (.qmd) jako čistý Quarto Markdown z hotových dat."""

    def __init__(self, model_name: str, global_rules: list[str] = None, task_rules: list[str] = None, forbidden_words: list[str] = None):
        forbidden_str = ", ".join([f"'{w}'" for w in (forbidden_words or [])]) if forbidden_words else "žádná"
        combined_rules = (global_rules or []) + (task_rules or [])
        all_rules = "\n".join([f"- {r}" for r in combined_rules])

        sys_msg = f"""Jsi odborný autor laboratorních protokolů v systému Quarto (.qmd) na VUT FEKT.
        Veškeré výpočty, grafy a analýzu fotografií už za tebe připravili předchozí agenti!
        Tvým úkolem je napsat kompletní tělo souboru protokol.qmd (povinné hlavní sekce: '# Teoretický úvod', '# Metodika', '# Výsledky', '# Diskuze', '# Závěr').

        ===================================================================
        1. KRITICKÁ PRAVIDLA PRO STRUKTURU A SYNTAXI (.qmd):
        ===================================================================
        - ŽÁDNÁ YAML HLAVIČKA: Nikdy na začátek dokumentu nepiš YAML hlavičku (žádné `---`, `title:`, `author:`). Začni rovnou na prvním řádku nadpisem `# Teoretický úvod`.
        - ČISTÝ MARKDOWN BEZ PYTHON BLOKŮ: Do dokumentu už NEPIŠ ŽÁDNÉ spustitelné bloky ```` ```{{python}} ````! Všechna vypočtená čísla ze `summary.json` i popisy fotek vidíš přímo ve schváleném plánu – vypiš tabulku přímo jako čistou Markdown tabulku (`| Sloupec 1 | Sloupec 2 |...`) a konkrétní čísla z ní rovnou dosaď přímo do textu v sekcích '# Diskuze' a '# Závěr'.
        - (Pojistka: Pokud bys z jakéhokoliv důvodu přesto použil Python blok s `display(Markdown(...))`, vždy použij trojité uvozovky `rf\"\"\"...\"\"\"`, nikdy nerozděluj jednořádkový řetězec klávesou Enter a na začátku bloku uveď `from IPython.display import display, Markdown`).
        - ZNAČKA NEJISTOTY: Značku `[NEJISTOTA: ...]` použij MAXIMÁLNĚ 1x v celém dokumentu, a to pouze tehdy, pokud v podkladech vysloveně chybí klíčový parametr. Nikdy nepoužívej více značek v jedné větě!

        ===================================================================
        2. JAK PRACOVAT PODLE TYPU ÚLOHY (VARIANTA A / B / C):
        ===================================================================
        VARIANTA A) POUZE NUMERICKÁ DATA (máš `summary.json` a grafy `processed/*.pdf`, žádné fotky v `img/`):
        - V sekci '# Metodika' popiš měřicí metodu a postup výpočtů.
        - V sekci '# Výsledky' vypiš souhrnnou Markdown tabulku ze `summary.json` a vlož všechny vygenerované grafy z `processed/*.pdf`.
        - V sekcích '# Diskuze' a '# Závěr' zhodnoť přesné číselné rozsahy a fyzikální souvislosti.

        VARIANTA B) PRAKTICKÁ / FOTOGRAFICKÁ ÚLOHA (žádná číselná data v `data/`, ale máš schválené kroky s fotkami):
        - NIKDY do textu nepiš, že soubor `processed/summary.json` nebyl nalezen nebo že chybí numerická data!
        - V sekci '# Metodika' napiš pouze 1 až 2 souvislé shrnující odstavce o použitých přístrojích a principu technologie (BEZ podkapitol a BEZ obrázků). Veškeré podkapitoly s postupem práce a fotografiemi vkládej VÝHRADNĚ do sekce '# Výsledky' (kde bude vždy odstavec vysvětlující postup a HNED pod ním příslušné fotografie)!
        - V sekci '# Výsledky' rozděl text do logických podkapitol (`## ...`) přesně podle `schvalene_kroky` v plánu.
        - NIKDY nevytvářej samostatnou podkapitolu `##` pro každou fotku zvlášť! Seskup fotografie do hlavních technologických podkapitol a do názvů nadpisů nepiš čísla ze zadání (jako 3.2.2), protože LaTeX nadpisy čísluje sám.
        - Ke každému kroku napiš souvislý odborný popis činnosti, v textu se odkaž na číslo obrázku v závorce `(Obr. X)` nebo `viz Obr. X` a pod odstavec vlož příslušné fotografie.
        - V sekcích '# Diskuze' a '# Závěr' zhodnoť praktický průběh práce, kritické technologické kroky a navrhni možná vylepšení pro průmyslovou praxi.

        VARIANTA C) KOMBINOVANÁ ÚLOHA (máš NUMERICKÁ DATA / GRAFY i FOTOGRAFIE):
        - V sekci '# Metodika' (případně v první části '# Výsledky') popiš praktický postup měření/výroby doplněný o všechny schválené fotografie s odkazy `(Obr. X)`.
        - V sekci '# Výsledky' dále uveď souhrnnou Markdown tabulku naměřených/vypočtených hodnot ze `summary.json` a vlož všechny analytické grafy z `processed/*.pdf`.
        - V sekcích '# Diskuze' a '# Závěr' propoj zhodnocení praktické realizace s konkrétními naměřenými čísly z tabulky a grafů.

        ===================================================================
        3. STRIKTNÍ PRAVIDLA PRO VKLÁDÁNÍ GRAFŮ A FOTOGRAFIÍ:
        ===================================================================
        - Musíš vložit VŠECHNY vygenerované grafy (`processed/*.pdf`) a VŠECHNY schválené fotografie v jejich přesném pořadí!
        - Každý graf nebo fotografii vlož VŽDY na samostatný řádek s PRÁZDNÝM ŘÁDKEM NAD I POD (jinak se v PDF slijí vedle sebe a ztratí popisky!):
        ![Přesný text popisku bez prefixu Obr. X](cesta/k/souboru){{width=65% fig-align="center"}}
        - ZÁKAZ DUPLICITY ČÍSLOVÁNÍ: Dovnitř hranatých závorek `![...]` NIKDY nepiš prefix `Obr. 1.` ani obecné názvy jako `Fotografie 1`! LaTeX čísluje obrázky jako `Obr. X.` zcela automaticky. Vždy tam vlož čistý odborný text z pole `caption`!
        - ZAKÁZANÁ SLOVA: Nikdy v celém protokolu (ani v Teoretickém úvodu!) nepoužívej tato slova: {forbidden_str}!
        {all_rules}"""
        super().__init__(model_name=model_name, system_prompt=sys_msg, temperature=0.2)

    def execute(
        self,
        verified_plan: dict,
        computed_summary: str,
        generated_plots: list[str],
        draft_file: Path,
    ) -> str:
        # Vytáhneme seznam všech schválených cest k fotkám přímo z verified_plan
        valid_img_paths = []
        for step in verified_plan.get("schvalene_kroky", []):
            valid_img_paths.extend(step.get("prirazene_fotky", []))

        prompt = (
            f"SCHVÁLENÝ PLÁN PROTOKOLU, TEORIE A KATALOG FOTEK (piš PŘÍSNĚ jen podle těchto bodů):\n"
            f"{json.dumps(verified_plan, ensure_ascii=False, indent=2)}\n\n"
            f"VÝSLEDKY NUMERICKÝCH VÝPOČTŮ:\n{computed_summary}\n\n"
            f"VYGENEROVANÉ GRAFY:\n{generated_plots}\n\n"
            f"POVINNÝ SEZNAM EXISTUJÍCÍCH SOUBORŮ FOTOGRAFIÍ (použij PŘESNĚ tyto cesty):\n"
            f"{valid_img_paths}\n\n"
            f"Sestav kompletní protokol od '# Teoretický úvod' až po '# Závěr' (BEZ jakýchkoliv ```{{python}} bloků!)."
        )

        raw_qmd = self.call_model(prompt)
        cleaned_qmd = self.strip_code_fences(raw_qmd, tag="qmd")
        cleaned_qmd = self.strip_code_fences(cleaned_qmd, tag="markdown")

        draft_file.parent.mkdir(parents=True, exist_ok=True)
        draft_file.write_text(cleaned_qmd, encoding="utf-8")
        print(f" [Quarto_Writer] Návrh uložen do '{draft_file.name}'.")
        return cleaned_qmd


class ProtocolInspectorAgent(BaseLabAgent):
    """Fáze 3: Opravuje chyby ze Scoreboardu a řeší značky [NEJISTOTA: ...] v návrhu."""

    def __init__(self, model_name: str, global_rules: list[str] = None):
        rules_str = "\n".join([f"- {r}" for r in (global_rules or [])])
        sys_msg = f"""Jsi přísný kontrolor laboratorních protokolů VUT FEKT (Protocol Inspector).
        Dostaneš návrh protokolu (.qmd), seznam syntaktických chyb ze Scoreboardu a případné značky `[NEJISTOTA: ...]` od autora.
        Tvůj úkol:
        1. Oprav všechny nahlášené chyby ze Scoreboardu a dodrž přitom pravidla:
        {rules_str}
        2. Zkontroluj každou značku `[NEJISTOTA: ...]`. Pokud odpověď najdeš ve schváleném plánu nebo ve vypočtených datech, nahraď značku přesným faktem.
        3. Pokud informaci nelze z podkladů zjistit, nahraď ji výrazným upozorněním: `> ** K RUČNÍMU DOPLNĚNÍ:** [popis, co má uživatel doplnit]`.
        4. Vrať kompletní, vyčištěný dokument od '# Teoretický úvod' až po '# Závěr' (bez úvodních či závěrečných komentářů okolo)."""
        super().__init__(model_name=model_name, system_prompt=sys_msg, temperature=0.1)

    def execute(self, draft_qmd: str, penalties: list[str], verified_plan_str: str, computed_summary: str) -> str:
        penalties_str = "\n".join(penalties) if penalties else "Žádné syntaktické chyby."
        prompt = (
            f"PENALIZACE ZE SCOREBOARDU K OPRAVĚ:\n{penalties_str}\n\n"
            f"SCHVÁLENÝ PLÁN A PODKLADY PRO OVĚŘENÍ FAKTŮ:\n{verified_plan_str}\n\n"
            f"VYPOČTENÁ DATA:\n{computed_summary}\n\n"
            f"NÁVRH PROTOKOLU K REVIZI:\n{draft_qmd}"
        )
        raw_fixed = self.call_model(prompt)
        cleaned_fixed = self.strip_code_fences(raw_fixed, tag="qmd")
        return self.strip_code_fences(cleaned_fixed, tag="markdown")

class VisionCuratorAgent(BaseLabAgent):
    """
    Fáze 1B: Zpracuje fotografie ve složce img/, opraví jejich EXIF rotaci,
    pomocí Vision modelu rozpozná, co na nich je, a sestaví 'processed/images_catalog.json'.
    """

    def __init__(self, model_name: str, forbidden_words: list[str] = None, use_ocr: bool = True):
        super().__init__(model_name=model_name, temperature=0.1)
        self.forbidden_words = forbidden_words or []
        self.reader = easyocr.Reader(["cs", "en"], gpu=False) if use_ocr else None

    def _extract_ocr_hints(self, img_path: str) -> str:
        """Lokálně (zdarma) přečte texty na displejích a štítcích přístrojů."""
        if not self.reader:
            return ""
        results = self.reader.readtext(img_path, detail=0)
        # Odfiltrujeme náhodný šum (1-2 znaky)
        clean_texts = [t.strip() for t in results if len(t.strip()) > 2]
        return ", ".join(clean_texts)

    #@staticmethod
    def _prepare_and_encode_image(self, img_path: str, max_size: int = 1024) -> str:
        """Opraví EXIF otočení fotky z telefonu, zmenší rozlišení pro PDF/API a vrátí base64."""
        with Image.open(img_path) as img:
            # 1. Oprava otočení z mobilu (EXIF Orientation tag)
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # 2. Zmenšení příliš velkých fotek (max 1400 px), aby PDF nemělo stovky MB
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def execute(self, image_files: list[str], verified_sections: list[str], task_rules: list[str]) -> list[dict]:
        catalog = []
        rules_str = "\n".join([f"- {r}" for r in (task_rules or [])])
        sections_str = "\n".join([f"- {s}" for s in verified_sections])
        forbidden_str = ", ".join([f"'{w}'" for w in self.forbidden_words]) if self.forbidden_words else "žádná"

        print(f" [Vision_Curator] Nalezeno {len(image_files)} obrázků ve složce 'img/'. Analyzuji obsah...")

        for idx, img_path in enumerate(image_files, start=1):
            ocr_hints = self._extract_ocr_hints(img_path)
            b64_img = self._prepare_and_encode_image(img_path)
            filename = os.path.basename(img_path)

            prompt = f"""Jsi odborný asistent v laboratořích VUT FEKT.
            Analyzuješ fotografii č. {idx} (soubor '{filename}') pro laboratorní protokol.

            POVOLENÉ SEKCE PROTOKOLU (vyber PŘESNĚ jednu z těchto sekcí, do které fotka patří):
            {sections_str}

            SPECIFICKÁ PRAVIDLA ÚLOHY:
            {rules_str}

            KRITICKÉ POKYNY PRO ANALÝZU OBRAZU:
            1. Popisuj VÝHRADNĚ to, co je na fotografii fyzicky vidět!
            2. Klicove slova a napisy na obrazku za tebe analyzovalo ocr, POUZIJ nasledujici slova: {ocr_hints}
            3. Zařaď fotografii do pole 'sekce' výběrem jedné položky ze seznamu POVOLENÉ SEKCE PROTOKOLU výše podle toho, co na fotce SKUTEČNĚ je.
            4. ZAKÁZANÁ SLOVA: Nikdy do políčka 'caption' ani 'postup_cinnosti' nevkládej tato slova: [{forbidden_str}]!

            Vrať ČISTÝ JSON objekt (bez ```json bloků) s těmito klíči:
            {{
            "obr_cislo": {idx},
            "file": "{img_path}",
            "sekce": "Přesný název vybrané sekce ze seznamu povolených sekcí výše",
            "caption": "Stručný, přesný odborný popisek pod obrázek (BEZ prefixu 'Obr. X.')",
            "postup_cinnosti": "2 až 3 odborné věty v trpném rodě popisující činnost a přístroje/hodnoty viditelné na fotce včetně odkazu '(Obr. {idx})'."
            }}"""
            try:
                # Volání přes LiteLLM (funguje pro Groq, Gemini, OpenAI i Claude)
                use_json_fmt = any(p in self.model_name.lower() for p in ["groq", "openai","gemini"])
                response = completion(
                    model=self.model_name,
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
                    max_tokens=2048,
                    reasoning_effort="none",
                    response_format={"type": "json_object"} if use_json_fmt else None,
                )
                raw_text = response.choices[0].message.content.strip()
                raw_json = self.strip_code_fences(raw_text, tag="json")
                item = json.loads(raw_json)
                # ESSENTIAL POJISTKA: Vždy vynutíme skutečné pořadové číslo a reálnou cestu k souboru
                item["obr_cislo"] = idx
                item["file"] = img_path
            except Exception as e:
                print(f"  [Vision_Curator] Záložní popisek pro {filename} ({e})")
                clean_name = os.path.splitext(filename)[0].replace("_", " ")
                fallback_section = verified_sections[0] if verified_sections else "Praktická část"
                item = {
                    "obr_cislo": idx,
                    "file": img_path,
                    "sekce": fallback_section,
                    "caption": f"Fotodokumentace kroku měření ({clean_name})",
                    "postup_cinnosti": f"Dokumentace pracovního postupu na pracovišti (Obr. {idx}).",
                }
            catalog.append(item)
            print(f"    ↳ Obr. {idx} ({filename}) [{item.get('sekce')}]: {item.get('caption')}")
        # ESSENTIAL: Uložení katalogu na disk
        os.makedirs("processed", exist_ok=True)
        with open("processed/images_catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

        return catalog