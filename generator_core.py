import os
import re
import json
import glob
import subprocess
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from agents import DataAnalystAgent, QuartoWriterAgent, ProtocolInspectorAgent

class BaseProtocolGenerator:
    """Base třída obsahující kompletní engine pro generování protokolů."""

    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir).resolve()
        self.root_dir = self.task_dir.parent.parent
        
        # Načtení .env z root složky
        load_dotenv(self.root_dir / ".env")
        
        self.config_list = [{
            "model": "openai/gpt-oss-120b",
            "api_type": "groq",
            "api_key": os.environ.get("GROQ_API_KEY"),
            "cache_seed": None,
        }]
        
        self.global_memory_path = self.root_dir / "global_memory.json"
        self.task_config_path = self.task_dir / "task_config.json"
        
        self.global_memory = self._load_global_memory()
        self.task_config = self._load_task_config()

    def _load_global_memory(self):
        base_global_rules = [
            "Při načítání dat MUSÍŠ použít PŘESNOU cestu k souboru ze složky 'data/' a PŘESNÉ názvy sloupců z přehledu.",
            "Aby Quarto vypsalo Markdown z Pythonu, přidej na začátek bloku '#| results: asis' a použij 'display(Markdown(rf\"...\"))'. Obyčejné display(df) je zakázané.",
            "Uvnitř Python f-stringů (rf\"...\") MUSÍŠ VŠECHNY LaTeXové složené závorky ZDVOJIT (např. $r_{{\\mathrm{{d}}}}$) nebo psát bez závorek ($r_d$).",
            "Pro matematiku používej VÝHRADNĚ dolary ($...$ a $$...$$). NIKDY nepoužívej \\(,\\), [, ].",
            "Před výpisem tabulky zaokrouhli hodnoty na 3 desetinná místa pomocí df = df.round(3).",
            "Sekce 'Diskuze' i 'Závěr' vypisuj přes display(Markdown(rf\"...\")) přímo z Pythonu s dosazenými reálnými hodnotami z tabulky!",
            "Sekce 'Závěr' NESMÍ obsahovat žádné odrážky ani číslování (1., 2.). Musí jít o souvislý odstavec textu.",
            "V Matplotlibu NIKDY nepoužívej zastaralý styl 'seaborn-whitegrid' (použij 'seaborn-v0_8-whitegrid')."
        ]
        if not self.global_memory_path.exists():
            mem = {"lessons_learned": base_global_rules, "gold_code_example": ""}
        else:
            with open(self.global_memory_path, "r", encoding="utf-8") as f:
                mem = json.load(f)
            for r in base_global_rules:
                if r not in mem["lessons_learned"]:
                    mem["lessons_learned"].append(r)

        with open(self.global_memory_path, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
        return mem

    def _load_task_config(self):
        with open(self.task_config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ensure_template_files(self):
            """
            Zkontroluje a případně vytvoří šablony v centrální složce root/template/.
            Nic nekopíruje do složky úlohy – vše se načítá centrálně!
            """
            template_dir = self.root_dir / "template"
            template_dir.mkdir(parents=True, exist_ok=True)

            # 1. Kontrola/vytvoření vut_preamble.tex v root/template/
            preamble_path = template_dir / "vut_preamble.tex"
            if not preamble_path.exists():
                preamble = (
                    "\\usepackage{ragged2e}\n"
                    "\\usepackage{setspace}\n"
                    "\\usepackage{float}\n"
                    "\\usepackage{booktabs}\n"
                    "\\usepackage{needspace}\n"
                    "\\usepackage{etoolbox}\n"
                    "\\preto{\\section}{\\needspace{6cm}}\n"
                    "\\providetoggle{nobreak}\n"
                    "\\appto{\\tabular}{\\toggletrue{nobreak}}\n"
                )
                with open(preamble_path, "w", encoding="utf-8") as f:
                    f.write(preamble)

            # 2. Automatické přejmenování vut_template.tex -> before-body.tex (Quarto vyžaduje tento název)
            old_title_path = template_dir / "vut_template.tex"
            title_path = template_dir / "before-body.tex"
            if old_title_path.exists() and not title_path.exists():
                old_title_path.rename(title_path)

            # 3. Kontrola/vytvoření before-body.tex s relativní cestou k logu ../../template/FEKT_logo.jpg
            title_code = r"""\begin{titlepage}
        \centering
        \IfFileExists{../../template/FEKT_logo.jpg}{
            \includegraphics[width=0.85\textwidth]{../../template/FEKT_logo.jpg}\par\vspace{1cm}
        }{
            {\scshape\Large Vysoké učení technické v Brně \par}
            \vspace{0.2cm}
            {\scshape\large Fakulta elektrotechniky a komunikačních technologií \par}
            \vspace{0.4cm}
            \hrule
            \vspace{1.2cm}
        }
        {\scshape\LARGE $predmet$ \par}
        \vspace{0.5cm}
        {\scshape\Large $kod_predmetu$ \par}
        \vspace{1.5cm}
        {\huge\bfseries Laboratorní úloha č. $cislo_ulohy$ \par}
        \vspace{2.5cm}
        {\huge\bfseries $title$ \par}
        \vfill
        \begin{flushleft}
            \large
            \textbf{Vypracoval:} $author$ ($student_id$) \hfill \textbf{Datum měření:} $datum_mereni$ \\
            $if(spolupracoval)$
            \textbf{Spolupracoval:} $spolupracoval$ \hfill \textbf{Datum odevzdání:} \today
            $endif$
        \end{flushleft}
    \end{titlepage}
    \newpage
    \justifying
    \onehalfspacing
    """
            with open(title_path, "w", encoding="utf-8") as f:
                f.write(title_code)

    def load_lab_context(self):
        """Načte zadani.pdf a všechny soubory v data/ konkrétní úlohy."""
        os.chdir(self.task_dir) # Přepnutí pracovního adresáře do složky úlohy
        meta = self.task_config["lab_metadata"]
        pdf_path = "zadani.pdf"
        
        zadani_text = ""
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            s_idx = (meta.get("start_page", 1) - 1)
            e_idx = meta.get("end_page", len(reader.pages))
            for i in range(s_idx, min(e_idx, len(reader.pages))):
                zadani_text += f"\n--- Strana {i+1} ---\n" + (reader.pages[i].extract_text() or "")
        else:
            print("⚠️ Upozornění: zadani.pdf nebylo ve složce úlohy nalezeno.")

        data_files = sorted(
            glob.glob("data/*.xlsx") + glob.glob("data/*.xls") + glob.glob("data/*.csv")
        )
        valid_files = [f.replace("\\", "/") for f in data_files]
        data_info_parts = []

        for fpath in valid_files:
            if fpath.endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(fpath)
                data_info_parts.append(f"SOUBOR: '{fpath}' | LISTY (sheets): {xls.sheet_names}")
                for sheet in xls.sheet_names:
                    # Načteme surovou mřížku bez headeru, aby agent viděl přesné indexy řádků a sloupců!
                    raw_df = pd.read_excel(fpath, sheet_name=sheet, header=None, nrows=10)
                    raw_df = raw_df.dropna(how="all", axis=0).dropna(how="all", axis=1)
                    data_info_parts.append(
                        f"--- Ukázka surových buněk listu '{sheet}' (header=None, prvních 10 řádků) ---\n"
                        f"{raw_df.to_string()}\n"
                    )
            elif fpath.endswith(".csv"):
                raw_df = pd.read_csv(fpath, header=None, nrows=10, sep=None, engine="python")
                data_info_parts.append(
                    f"SOUBOR CSV: '{fpath}' (prvních 10 řádků, header=None):\n{raw_df.to_string()}\n"
                )

        data_info = "\n".join(data_info_parts)
        return zadani_text, data_info, valid_files

    def custom_scoreboard_checks(self, qmd_text: str) -> tuple[int, list[str]]:
        """
        Abstraktní/virtuální metoda pro specifické kontroly konkrétní úlohy.
        Může být přepsána (overridden) v podtřídě run_task.py!
        """
        deduction = 0
        penalties = []
        for word in self.task_config.get("forbidden_words", []):
            if word.lower() in qmd_text.lower():
                deduction += 40
                penalties.append(f"-40 bodů: Použil jsi zakázané slovo/koncept pro tuto úlohu: '{word}'!")
        return deduction, penalties

    def evaluate_qmd_scoreboard(self, qmd_text: str, valid_files: list[str] = None):
        """
        Scoreboard pro Fázi 3: Kontroluje .qmd dokument (Quarto syntaxi, LaTeX, strukturu a text).
        """
        score = 100
        penalties = []

        # 1. Kontrola, zda Writer/Inspector vyřešil všechny značky [NEJISTOTA: ...]
        uncertainties = re.findall(r"\[NEJISTOTA:.*?\]", qmd_text)
        if uncertainties:
            score -= 30
            penalties.append(
                f"-30 bodů: [NEJISTOTA] V textu zůstaly nevyřešené značky {uncertainties}! "
                f"Ověř je ze zadání/dat a nahraď je buď faktem, nebo boxem '> **⚠️ K RUČNÍMU DOPLNĚNÍ:** ...'!"
            )

        # Kontrola, že agent nenačítá neexistující 'data/summary.json' místo 'processed/summary.json'
        if "data/summary.json" in qmd_text:
            score -= 50
            penalties.append("-50 bodů: Soubor summary.json leží ve složce 'processed/summary.json', nikoliv v 'data/summary.json'!")

        # 2. Kontrola Quarto bloků a výstupů
        if "```python" in qmd_text and "```{python}" not in qmd_text:
            score -= 50
            penalties.append("-50 bodů: Použil jsi neaktivní ```python místo spustitelného Quarto bloku ```{python}!")

        if "display(df)" in qmd_text:
            score -= 20
            penalties.append("-20 bodů: Zakázané display(df). Pro tabulku použij display(Markdown(df.to_markdown(index=False))).")

        if "Markdown(" in qmd_text and "display(Markdown" not in qmd_text:
            score -= 30
            penalties.append("-30 bodů: Použil jsi Markdown() bez obalení do display().")

        # 3. Kontrola syntaxe Markdownu a f-stringů mimo kód
        text_outside_code = re.sub(r"```\{python\}.*?```", "", qmd_text, flags=re.DOTALL)
        if re.search(r"\{[a-zA-Z0-9_]+:[.0-9]+f\}", text_outside_code):
            score -= 30
            penalties.append("-30 bodů: Mimo Python bloky f-stringy s formátováním '{promenna:.3f}' nefungují!")

        if "..." in qmd_text or "DOPLŇ" in qmd_text:
            score -= 30
            penalties.append("-30 bodů: V textu zůstaly výpustky '...' nebo 'DOPLŇ'.")

        # 4. Kontrola matematiky a LaTeX závorek
        if r"\[" in qmd_text or r"\]" in qmd_text or r"\(" in qmd_text or r"\)" in qmd_text:
            score -= 40
            penalties.append("-40 bodů: Pro matematiku používej VÝHRADNĚ dolary ($...$ nebo $$...$$), žádné \\( či \\[!")

        # 5. Kontrola sekce Závěr (zákaz seznamů)
        if "Závěr" in qmd_text:
            zaver_text = qmd_text.split("Závěr")[-1]
            if re.search(r"^\s*(?:[\*\-]|\d+\.)\s+", zaver_text, re.MULTILINE):
                score -= 40
                penalties.append("-40 bodů: Sekce Závěr nesmí obsahovat odrážky ani číslování, musí jít o souvislý odstavec!")
        # 4. Kontrola matematiky a LaTeX závorek (ignoruje LaTeX odřádkování \\[6pt])
        if re.search(r"(?<!\\)\\\[|(?<!\\)\\\]|(?<!\\)\\\(|(?<!\\)\\\)", qmd_text):
            score -= 40
            penalties.append(
                "-40 bodů: Pro matematiku používej VÝHRADNĚ dolary ($...$ nebo $$...$$), nikoliv LaTeX závorky \\( a \\[. "
                "(POZOR: Hranaté závorky v Python kódu jako df['sloupec'] jsou samozřejmě POVINNÉ a nesmíš je měnit!)"
            )
        # 6. Kontrola nezdvojených LaTeX závorek v Python blocích
        python_blocks = re.findall(r"```\{python\}.*?```", qmd_text, flags=re.DOTALL)
        for block in python_blocks:
            # Hledá jednoduchou závorku u LaTeX příkazu nebo u indexu obsahujícího LaTeX příkaz
            if re.search(r"\\[a-zA-Z]+\{(?!\{)|[_^]\{(?!\{)[^{}]*\\[a-zA-Z]+", block):
                score -= 40
                penalties.append(
                    "-40 bodů: CHYBA SYNTAXE! Uvnitř Python f-stringu jsi použil jednoduché složené závorky "
                    "u LaTeXu (např. $I_{\\mathrm{{S}}}$). I vnější závorku za podtržítkem MUSÍŠ ZDVOJIT: $I_{{\\mathrm{{S}}}}$!"
                )
                break
        # 7. Kontrola kompletnosti dokumentu a uzavření bloků
        all_fences = len(re.findall(r"```", qmd_text))
        if all_fences % 2 != 0:
            score -= 80
            penalties.append("-80 bodů: KRITICKÁ CHYBA! Tvůj výstup je USEKNUTÝ uprostřed bloku (lichý počet ```)!")

        required_sections = ["Teoretický úvod", "Výsledky", "Diskuze", "Závěr"]
        missing_sections = [sec for sec in required_sections if sec.lower() not in qmd_text.lower()]
        if missing_sections:
            score -= 60
            penalties.append(f"-60 bodů: Dokument je NEKOMPLETNÍ! Chybí povinné sekce: {missing_sections}.")

        # 8. Zavolání specifických kontrol pro danou úlohu z run_task.py (např. forbidden_words)
        custom_deduction, custom_penalties = self.custom_scoreboard_checks(qmd_text)
        score -= custom_deduction
        penalties.extend(custom_penalties)

        return score, penalties

    def _inspect_and_refine(self, qmd_body: str, inspector, zadani_text: str, computed_summary: str, valid_files: list[str]) -> str:
        """
        Tvá původní while smyčka se sledováním progrese a detekcí zacyklení!
        Nyní místo Writera úkoluje ProtocolInspectorAgenta.
        """
        best_score = -999
        best_qmd = qmd_body
        iteration = 1
        max_iterations = 4
        penalty_history_count = {}

        while iteration <= max_iterations:
            qmd_body = self._sanitize_output(qmd_body)
            score, penalties = self.evaluate_qmd_scoreboard(qmd_body, valid_files)
            print(f"\n📊 SCOREBOARD (Iterace {iteration}): {score}/100 bodů")

            if score == 100:
                print("  ✅ Dosaženo 100/100 bodů! Návrh je bezchybný.")
                best_qmd = qmd_body
                break

            # Ukládáme si pouze skutečné zlepšení
            if score > best_score:
                best_score = score
                best_qmd = qmd_body

            # Detekce zacyklení (pokud stejná chyba vyskočí 2x po sobě)
            stuck = False
            for p in penalties:
                p_key = p[:45]
                penalty_history_count[p_key] = penalty_history_count.get(p_key, 0) + 1
                if penalty_history_count[p_key] >= 2:
                    print(f"  🛑 ZACYKLENÍ DETEKOVÁNO: Chyba '{p_key}...' se opakuje už 2x za sebou!")
                    stuck = True

            if stuck or iteration == max_iterations:
                print(f"  ⚠️ Ukončuji smyčku. Používám nejlepší nalezenou verzi se skóre {best_score}/100 bodů.")
                break

            print("  ⚠️ Zjištěny prohřešky nebo nejistoty, předávám k opravě Protocol_Inspectorovi:")
            for p in penalties:
                print("    ", p)

            # Zavoláme ProtocolInspectorAgenta, aby chyby a [NEJISTOTA: ...] opravil
            candidate_body = inspector.execute(qmd_body, penalties, zadani_text, computed_summary)

            if len(candidate_body) > 200:
                qmd_body = candidate_body
            else:
                print("  ⚠️ Inspector vrátil prázdnou/krátkou odpověď, ponechávám předchozí verzi.")

            iteration += 1

        # Na závěr zkontrolujeme finální stav nejlepší verze, zapíšeme případný audit_protokolu.txt a očistíme LaTeX
        _, final_penalties = self.evaluate_qmd_scoreboard(best_qmd, valid_files)
        annotated_qmd = self.annotate_and_log_issues(best_qmd, final_penalties)
        return self._sanitize_output(annotated_qmd)

    def build_yaml_header(self):
        cfg = self.task_config["lab_metadata"]
        yaml_lines = [
            "---",
            f'title: "{cfg["title"]}"',
            f'author: "{cfg["author"]}"',
            f'student_id: "{cfg["student_id"]}"',
            f'predmet: "{cfg["predmet"]}"',
            f'kod_predmetu: "{cfg["kod_predmetu"]}"',
            f'cislo_ulohy: "{cfg["cislo_ulohy"]}"',
            f'datum_mereni: "{cfg["datum_mereni"]}"',
            f'spolupracoval: "{cfg["spolupracoval"]}"',
            "lang: cs",
            "format:",
            "  pdf:",
            "    documentclass: article",
            "    papersize: a4",
            "    fontsize: 12pt",
            "    geometry:",
            "      - margin=2.5cm",
            "    number-sections: true",
            "    keep-tex: true",
            "    include-in-header: ../../template/vut_preamble.tex",
            "    template-partials:",
            "      - ../../template/before-body.tex",
            "execute:",
            "  echo: false",
            "  warning: false",
            "  message: false",
            "---",
            "", # Garantuje prázdný řádek za uzavíracím ---
            ""
        ]
        return "\n".join(yaml_lines)
    
    def run(self, force_stage: int = 0):
        """Hlavní orchestrátor: připraví prostředí a řídí jednotlivé fáze pipeline."""
        os.chdir(self.task_dir)
        self.ensure_template_files()
        zadani_text, data_info, valid_files = self.load_lab_context()

        processed_dir = self.task_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        summary_file = processed_dir / "summary.json"
        draft_file = self.task_dir / "draft_protokol.qmd"
        final_file = self.task_dir / "protokol.qmd"

        global_rules = self.global_memory["lessons_learned"]
        task_rules = self.task_config.get("task_specific_rules", [])

        ran_previous_stage = False
        # --- FÁZE 1: ZPRACOVÁNÍ DAT A GRAFŮ ---
        if not summary_file.exists() or force_stage == 1:
            print("\n🚀 [FÁZE 1/4] Spouštím DataAnalystAgent...")
            analyst = DataAnalystAgent(self.config_list, valid_files, task_rules)
            analyst.execute(zadani_text, data_info)
            ran_previous_stage = True
        else:
            print("\n⏩ [FÁZE 1/4] Přeskakuji (nalezen 'processed/summary.json').")

        with open(summary_file, "r", encoding="utf-8") as f:
            computed_summary = f.read()
        generated_plots = sorted([p.replace("\\", "/") for p in glob.glob("processed/*.pdf")])

        # --- FÁZE 2: NÁVRH PROTOKOLU ---
        if not draft_file.exists() or ran_previous_stage or force_stage in [1, 2]:
            print("\n✍️ [FÁZE 2/4] Spouštím QuartoWriterAgent...")
            writer = QuartoWriterAgent(self.config_list, global_rules, task_rules)
            writer.execute(zadani_text, computed_summary, generated_plots, draft_file)
            ran_previous_stage = True
        else:
            print("\n⏩ [FÁZE 2/4] Přeskakuji (nalezen 'draft_protokol.qmd').")

        with open(draft_file, "r", encoding="utf-8") as f:
            current_qmd = f.read()

        # --- FÁZE 3: INSPEKCE A SCOREBOARD SMYČKA ---
        if not final_file.exists() or ran_previous_stage or force_stage in [1, 2, 3]:
            print("\n🕵️ [FÁZE 3/4] Spouštím kontrolu kvality (Scoreboard + ProtocolInspectorAgent)...")
            inspector = ProtocolInspectorAgent(self.config_list, global_rules)
            current_qmd = self._inspect_and_refine(current_qmd, inspector, zadani_text, computed_summary, valid_files)
            
            with open(final_file, "w", encoding="utf-8") as f:
                f.write(self.build_yaml_header() + current_qmd)
        else:
            print("\n⏩ [FÁZE 3/4] Přeskakuji (nalezen hotový 'protokol.qmd').")

        # --- FÁZE 4: KOMPILACE DO PDF ---
        print("\n🖨️ [FÁZE 4/4] Spouštím překlad v Quartu...")
        self.compile_pdf()

    def _clean_fences(self, text: str) -> str:
        """Odstraní vnější markdown blok POUZE tehdy, pokud jím agent obalil celý výstup."""
        text = text.strip()
        for prefix in ["```markdown", "```quarto", "```qmd"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                return text
        return text

    def _sanitize_output(self, text: str) -> str:
        if "# Teoretický úvod" in text:
            text = text[text.index("# Teoretický úvod"):]
        # 0b. Zajištění prázdného řádku mezi po sobě jdoucími obrázky ![...](...), aby se v PDF nepřekrývaly vedle sebe!
        text = re.sub(r"(\!\[.*?\]\(.*?\)(?:\{.*?\})?)\s*\n\s*(\!\[)", r"\1\n\n\2", text)
        # 1. Automatická oprava ```python na spustitelný Quarto blok ```{python}
        text = re.sub(r"```python\b", "```{python}", text)
        # 2. Automatická oprava cesty k summary.json
        text = text.replace("data/summary.json", "processed/summary.json")
        # 3. Oprava případného poškození Python indexování (pokud agent přepsal [ ] na $$)
        text = re.sub(r"df\$\$(.*?)\$\$", r"df[\1]", text)
        text = re.sub(r"\.iloc\$\$(\d+)\$\$", r".iloc[\1]", text)
        text = re.sub(r"(task\d+)\$\$'(.*?)'\$\$", r"\1['\2']", text)
        # 4. Automatická oprava LaTeX závorek \(...\) a [...] POUZE mimo Python bloky (nebo bezpečně s негативním lookbehindem pro \\)
        text = re.sub(r"(?<!\\)\\\[", "$$", text) 
        text = re.sub(r"(?<!\\)\\\]", "$$", text)
        text = re.sub(r"(?<!\\)\\\(", "$", text) 
        text = re.sub(r"(?<!\\)\\\)", "$", text)
        # 5. Očištění problematických Unicode znaků
        text = text.replace("\u2011", "-").replace("\\u2011", "-").replace("\u202f", " ").replace("\\u202f", " ")
        # 6. NEPRŮSTŘELNÁ OPRAVA LaTeX závorek uvnitř ```{python} bloků (zvládá i vnořené _{ \text{...} })
        def fix_python_block(match):
            block = match.group(0)
            # A) Nejprve zdvojíme závorky přímo za LaTeX příkazy: \text{low} -> \text{{low}}, \mathrm{S} -> \mathrm{{S}}
            block = re.sub(r"(\\[a-zA-Z]+)\{(?!\{)([^{}]*?)(?<!\})\}", r"\1{{\2}}", block)
            # B) Poté zdvojíme závorky u dolních/horních indexů, které obsahují LaTeX příkaz: _{\mathrm{{S}}} -> _{{\mathrm{{S}}}}
            block = re.sub(r"([_^])\{(?!\{)([^{}]*?(?:\{\{[^{}]*?\}\}[^{}]*?)*\\[a-zA-Z]+[^{}]*?)(?<!\})\}", r"\1{{\2}}", block)
            return block

        text = re.sub(r"```\{python\}.*?```", fix_python_block, text, flags=re.DOTALL)
        return text

    def annotate_and_log_issues(self, qmd_body: str, penalties: list[str]) -> str:
            """Pokud zůstaly v kódu prohřešky, najde přesné řádky, vypíše je a označí v .qmd."""
            if not penalties:
                return qmd_body
    
            lines = qmd_body.splitlines()
            report_lines = ["\n" + "="*60, "🔍 DIAGNOSTICKÝ AUDIT NEVYŘEŠENÝCH PROHŘEŠKŮ (K RUČNÍ KONTROLE):", "="*60]
    
            for p in penalties:
                report_lines.append(f"• {p}")
                # Pokusíme se najít číslo řádku podle typu chyby
                for idx, line in enumerate(lines, start=1):
                    if ("seaborn" in p and "seaborn" in line) or \
                       ("savefig" in p and ("savefig" in line or "close" in line)) or \
                       ("LaTeX závorky" in p and re.search(r"\\[a-zA-Z]+\{[^{}]+\}", line)) or \
                       ("výpustky" in p and ("..." in line or "DOPLŇ" in line)):
                        report_lines.append(f"    ↳ Nalezeno na řádku {idx}: {line.strip()}")
                        lines[idx - 1] = line + "  # <-- ⚠️ ZDE ZKONTROLOVAT (SCOREBOARD)"
    
            report_lines.append("="*60 + "\n")
            audit_text = "\n".join(report_lines)
            print(audit_text)
    
            with open("audit_protokolu.txt", "w", encoding="utf-8") as f:
                f.write(audit_text)
    
            return "\n".join(lines)

    def compile_pdf(self):
        print("\nSpouštím překlad v Quartu...")
        res = subprocess.run(["quarto", "render", "protokol.qmd", "--to", "pdf"], capture_output=True, text=True, encoding="utf-8", shell=True)
        if res.returncode == 0:
            print("🎉 ÚSPĚCH! Otevírám protokol.pdf...")
            os.startfile("protokol.pdf")
        else:
            print("❌ CHYBA KOMPILACE:\n", res.stderr)