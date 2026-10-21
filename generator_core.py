import os
import re
import json
import glob
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from agents import DataAnalystAgent, QuartoWriterAgent, ProtocolInspectorAgent
from autogen import ConversableAgent, UserProxyAgent

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
        
        md_zadani = ""
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            s_idx = (meta.get("start_page", 1) - 1)
            e_idx = meta.get("end_page", len(reader.pages))
            for i in range(s_idx, min(e_idx, len(reader.pages))):
                md_zadani += f"\n--- Strana {i+1} ---\n" + (reader.pages[i].extract_text() or "")
        else:
            print("⚠️ Upozornění: zadani.pdf nebylo ve složce úlohy nalezeno.")

        data_info = ""
        valid_files = []

        for xlsx_path in glob.glob("data/*.xlsx"):
            xlsx_path = xlsx_path.replace("\\", "/")
            valid_files.append(xlsx_path)
            all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
            data_info += f"\n=== EXCEL: '{xlsx_path}' (Listy: {list(all_sheets.keys())}) ===\n"
            for s_name, df in all_sheets.items():
                df = df.dropna(how="all")
                data_info += f"  -> List '{s_name}' | Sloupce: {list(df.columns)}\n{df.head(3).to_string()}\n"

        for csv_path in glob.glob("data/*.csv"):
            csv_path = csv_path.replace("\\", "/")
            valid_files.append(csv_path)
            df = pd.read_csv(csv_path)
            data_info += f"\n=== CSV: '{csv_path}' | Sloupce: {list(df.columns)} ===\n{df.head(3).to_string()}\n"

        if not valid_files:
            raise FileNotFoundError("❌ USER INPUT ERROR: Složka 'data/' je prázdná! Vlož do ní naměřená data (.xlsx nebo .csv).")

        return md_zadani, data_info, valid_files

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

    def evaluate_qmd_scoreboard(self, qmd_text: str, valid_files: list[str]):
        """Hlavní Scoreboard kombinující globální kontroly a specifické kontroly úlohy."""
        score = 100
        penalties = []

        has_read_call = bool(re.search(r"read_(?:excel|csv)\s*\(", qmd_text))
        used_valid_file = any(vf in qmd_text.replace("\\", "/") for vf in valid_files)

        if not has_read_call or not used_valid_file:
            score -= 60
            penalties.append(
                f"-60 bodů: [SOUBOR] V kódu chybí načtení správného souboru přes pd.read_excel/read_csv! "
                f"Musíš v kódu použít přesně tuto cestu: {valid_files}"
            )

        # 8. KONTROLA KOMPLETNOSTI DOKUMENTU A UZAVŘENÍ BLOKŮ (Ochrana proti useknutému výstupu)
        open_blocks = len(re.findall(r"```\{python\}", qmd_text))
        all_fences = len(re.findall(r"```", qmd_text))
        # Každý otevřený blok musí mít svůj uzavírací ``` (celkový počet ``` musí být sudý a >= 2 * open_blocks)
        if open_blocks == 0 or all_fences % 2 != 0:
            score -= 80
            penalties.append(
                "-80 bodů: KRITICKÁ CHYBA! Tvůj výstup je USEKNUTÝ uprostřed Python bloku (chybí uzavírací ```)! "
                "Nevypisuj v cyklu celé tabulky měření pro každou diodu zvlášť, piš stručněji a dokonči celý dokument!"
            )

        required_sections = ["Teoretický úvod", "Výsledky", "Diskuze", "Závěr"]
        missing_sections = [sec for sec in required_sections if sec.lower() not in qmd_text.lower()]
        if missing_sections:
            score -= 60
            penalties.append(
                f"-60 bodů: Dokument je NEKOMPLETNÍ! Chybí v něm tyto povinné kapitoly: {missing_sections}. "
                f"Dokument musí obsahovat: Teoretický úvod, Metodiku, Výsledky, Diskuzi a Závěr!"
            )

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

        # 6. Kontrola Matplotlibu, stylů a grafů
        if "seaborn-whitegrid" in qmd_text or "style.use('seaborn')" in qmd_text or "import seaborn" in qmd_text:
            score -= 30
            penalties.append("-30 bodů: Nepoužívej seaborn! Nastavuj styl čistého Matplotlibu přes plt.style.use('seaborn-v0_8-whitegrid').")

        if "plt.savefig" in qmd_text or "plt.close" in qmd_text:
            score -= 40
            penalties.append("-40 bodů: Nepoužívej plt.savefig() ani plt.close()! Quarto vkládá grafy automaticky přes plt.show().")

        if "subplots(" in qmd_text and ("3, 4" in qmd_text or "4, 3" in qmd_text or "len(sheets)" in qmd_text):
            score -= 40
            penalties.append("-40 bodů: Nevykresluj všechny grafy najednou do jedné obří mřížky!")

        # 7. Kontrola raw f-stringů a zdvojených složených závorek v Pythonu
        if 'f"' in qmd_text and '\\' in qmd_text and 'rf"' not in qmd_text and "rf'" not in qmd_text:
            score -= 30
            penalties.append("-30 bodů: Pokud řetězec obsahuje LaTeXová lomítka, použij raw f-string (předponu rf\"...\").")

        if "r'I_{{" in qmd_text or "r'r_{{" in qmd_text or "r'U_{{" in qmd_text:
            score -= 40
            penalties.append("-40 bodů: Nedávej LaTeXové složené závorky do názvů sloupců tabulky!")

        python_blocks = re.findall(r"```\{python\}.*?```", qmd_text, flags=re.DOTALL)
        for block in python_blocks:
            if re.search(r"(\\[a-zA-Z]+\{[^{}]+\}|_\{[^{}]*\\[a-zA-Z]+[^{}]*\})", block):
                score -= 40
                penalties.append(
                    "-40 bodů: CHYBA SYNTAXE! Uvnitř Python f-stringu (rf\"...\") jsi použil jednoduché LaTeX závorky. "
                    "Všechny LaTeXové závorky uvnitř f-stringů musíš zdvojit (např. $r_{{\\mathrm{{d}}}}$)!"
                )
                break

        # 3. Zavolání dědičné metody pro specifické kontroly dané úlohy!
        custom_deduction, custom_penalties = self.custom_scoreboard_checks(qmd_text)
        score -= custom_deduction
        penalties.extend(custom_penalties)

        return score, penalties

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
    
    def run(self):
        """Hlavní šablonová metoda (Template Method), která řídí celý proces."""
        os.chdir(self.task_dir)
        self.ensure_template_files()
        zadani_text, data_info, valid_files = self.load_lab_context()

        all_rules = self.global_memory["lessons_learned"] + self.task_config.get("task_specific_rules", [])
        rules_str = "\n".join([f"- {r}" for r in all_rules])

        writer_agent = ConversableAgent(
            name="Quarto_Writer",
            system_message=f"""Jsi expert na sázecí systém Quarto (.qmd) a Python na VUT FEKT.
            Sestav kompletní tělo laboratorního protokolu (BEZ YAML hlavičky, začni rovnou '# Teoretický úvod').
            Dokument MUSÍ obsahovat sekce: # Teoretický úvod, # Metodika, # Výsledky, # Diskuze a # Závěr.
            V cyklu pro jednotlivé listy Excelu vykresluj pouze grafy (NIKDY nevypisuj podrobné tabulky surových dat pro každý list zvlášť, vypiš jen jednu celkovou souhrnnou tabulku výsledků!).

            STRIKTNÍ PRAVIDLA (GLOBÁLNÍ + SPECIFICKÁ PRO TUTO ÚLOHU):
            {rules_str}""",
            # Zvýšení max_tokens na 8192, aby Groq nikdy neuřízl kód v polovině!
            llm_config={"config_list": self.config_list, "temperature": 0.1, "max_tokens": 8192},
        )
        user = UserProxyAgent(name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config=False)

        prompt = f"ZADÁNÍ:\n{zadani_text}\n\nDATA:\n{data_info}\n\nSestav kompletní tělo souboru protokol.qmd až po sekci # Závěr."
        res = user.initiate_chat(writer_agent, message=prompt)
        qmd_body = self._clean_fences(res.chat_history[-1]["content"].strip())

        best_score = -999
        best_qmd = qmd_body
        iteration = 1
        max_iterations = 4
        penalty_history_count = {}

        while iteration <= max_iterations:
            score, penalties = self.evaluate_qmd_scoreboard(qmd_body, valid_files)
            print(f"\nSCOREBOARD (Iterace {iteration}): {score}/100 bodů")

            if score == 100:
                print("Dosaženo 100/100 bodů!")
                best_qmd = qmd_body
                break

            # Ukládáme si pouze skutečné zlepšení
            if score > best_score:
                best_score = score
                best_qmd = qmd_body

            # Kontrola opakujících se identických chyb (použijeme prvních 45 znaků celé penalizace)
            stuck = False
            for p in penalties:
                p_key = p[:45]
                penalty_history_count[p_key] = penalty_history_count.get(p_key, 0) + 1
                if penalty_history_count[p_key] >= 2:
                    print(f"ZACYKLENÍ DETEKOVÁNO: Chyba '{p_key}...' se opakuje už 2x za sebou!")
                    stuck = True

            # Teď už break správně ukončí celou while smyčku a zachová předchozí 'best_qmd'!
            if stuck or iteration == max_iterations:
                print(f"Ukončuji smyčku. Používám nejlepší nalezenou verzi se skóre {best_score}/100 bodů.")
                break

            print("Zjištěny prohřešky, posílám agentovi penalizaci k opravě:")
            for p in penalties:
                print("  ", p)

            fix_prompt = (
                f"Tvůj předchozí výstup získal pouze {score}/100 bodů kvůli těmto chybám:\n"
                + "\n".join(penalties) +
                f"\n\nZde je tvůj předchozí kód, který musíš OPRAVIT a DOKONČIT:\n{qmd_body}\n\n"
                f"STRIKTNÍ PŘÍKAZ: Vrať KOMPLETNÍ opravený kód od '# Teoretický úvod' až do konce '# Závěr'. "
                f"Všechny Python bloky musí být řádně ukončeny značkou ```. Použij soubor {valid_files}."
            )
            
            res_fix = writer_agent.generate_reply(messages=[{"role": "user", "content": fix_prompt}])
            candidate_body = res_fix.get("content", "") if isinstance(res_fix, dict) else str(res_fix)
            candidate_body = self._clean_fences(candidate_body.strip())

            # Přijmeme nový kandidátní text, pokud není prázdný
            if len(candidate_body) > 200:
                qmd_body = candidate_body
            else:
                print("Agent vrátil prázdnou odpověď, ponechávám předchozí verzi.")

            iteration += 1

        # Použití nejlepší verze, sanitizace a uložení
        final_body = self._sanitize_output(best_qmd)
        with open("protokol.qmd", "w", encoding="utf-8") as f:
            f.write(self.build_yaml_header() + final_body)

        # Zjistíme finální stav nejlepší verze a případně označíme řádky k ruční kontrole
        score, final_penalties = self.evaluate_qmd_scoreboard(best_qmd, valid_files)
        annotated_qmd = self.annotate_and_log_issues(best_qmd, final_penalties)
        final_body = self._sanitize_output(annotated_qmd)
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
        text = text.replace("\u2011", "-").replace("\\u2011", "-").replace("\u202f", " ").replace("\\u202f", " ")
        text = re.sub(r"\\(mathrm|text|textbf|mathit)\{([^{}]+)\}", r"\\\1{{\2}}", text)
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