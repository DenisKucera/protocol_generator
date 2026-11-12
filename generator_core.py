import os
import re
import json
import glob
import subprocess
import pandas as pd
from PIL import Image
from pathlib import Path
from pypdf import PdfReader
#from dotenv import load_dotenv
from agents import DataAnalystAgent, QuartoWriterAgent, ProtocolInspectorAgent, VisionCuratorAgent, TaskPlannerAgent
from checkers import MaterialVerifier, QualityControl
from model_config import ModelConfig

class BaseProtocolGenerator:
    """Base třída obsahující kompletní engine pro generování protokolů."""

    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir).resolve()
        self.root_dir = self.task_dir.parent.parent
        
        self.models = ModelConfig(env_path=self.root_dir / ".env")
        
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

    def _get_sorted_images(self) -> list[str]:
        """
        Najde všechny obrázky v 'img/' a seřadí je chronologicky podle EXIF data pořízení
        (DateTimeOriginal). Pokud fotka EXIF nemá, použije čas změny souboru a název.
        """
        raw_files = [
            p.replace("\\", "/")
            for p in glob.glob("img/*.jpg") + glob.glob("img/*.jpeg") + glob.glob("img/*.png")
            if "FEKT_logo" not in p
        ]

        def get_image_timestamp(img_path: str):
            try:
                with Image.open(img_path) as img:
                    exif = img.getexif()
                    if exif:
                        # EXIF 36867 = DateTimeOriginal, 306 = DateTime (formát 'YYYY:MM:DD HH:MM:SS')
                        dt_str = exif.get(36867) or exif.get(306)
                        if dt_str:
                            return (0, str(dt_str), img_path)
            except Exception:
                pass
            # Záložní řazení: pokud byly soubory pojmenovány číselně (např. 1.jpg, 10.jpg), řadíme přirozeně podle čísla
            nums = re.findall(r"\d+", os.path.basename(img_path))
            num_key = int(nums[0]) if nums else 999999
            return (1, f"{num_key:08d}", img_path)

        return [f for f in sorted(raw_files, key=get_image_timestamp)]

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
            print(" Upozornění: zadani.pdf nebylo ve složce úlohy nalezeno.")

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
        Implementace je v run_task.py!
        """
        return 0, []

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
        forbidden_words = self.task_config.get("forbidden_words", [])
        plan_file = processed_dir / "task_plan.json"

        # --- FÁZE 0: EXTRAKCE PLÁNU ZE ZADÁNÍ (Plan-and-Execute) ---
        if not plan_file.exists() or force_stage == 1:
            print("\n [FÁZE 0/4] Spouštím TaskPlannerAgent (extrakce požadavků ze zadání)...")
            planner = TaskPlannerAgent(model_name=self.models.planner_model, task_rules=task_rules, forbidden_words=forbidden_words)
            plan_data = planner.execute(zadani_text, plan_file)
            ran_previous_stage = True
        else:
            print("\n [FÁZE 0/4] Přeskakuji (nalezen 'processed/task_plan.json').")
            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)

        # Vytáhneme čistý seznam názvů kapitol pro VisionCuratorAgenta
        verified_sections = [krok.get("nazev_sekce", "") for krok in plan_data.get("pozadovane_kroky", [])]

        ran_previous_stage = False
        # --- FÁZE 1A: ZPRACOVÁNÍ NUMERICKÝCH DAT (Pokud existuje Excel/CSV v data/) ---
        if valid_files:
            if not summary_file.exists() or force_stage == 1:
                print("\n [FÁZE 1A/4] Spouštím DataAnalystAgent (numerická data)...")
                analyst = DataAnalystAgent(self.models.analyst_model, valid_files, task_rules)
                analyst.execute(plan_data, data_info)
                ran_previous_stage = True
            else:
                print("\n [FÁZE 1A/4] Přeskakuji (nalezen 'processed/summary.json').")
            with open(summary_file, "r", encoding="utf-8") as f:
                computed_summary = f.read()
        else:
            print("\n [FÁZE 1A/4] Ve složce 'data/' nejsou žádná číselná data (praktická/fotografická úloha).")
            computed_summary = "Tato úloha neobsahuje číselné tabulky v data/, jde o praktickou technologickou úlohu."
        generated_plots = sorted([p.replace("\\", "/") for p in glob.glob("processed/*.pdf")])
        # --- FÁZE 1B: ZPRACOVÁNÍ FOTOGRAFIÍ (Pokud jsou fotky ve složce img/) ---
        image_files = self._get_sorted_images()
        catalog_file = processed_dir / "images_catalog.json"

        if image_files:
            need_vision_run = not catalog_file.exists() or force_stage == 1

            # Pokud katalog existuje, zkontrolujeme, zda přesně odpovídá aktuálním souborům v img/
            if catalog_file.exists() and not need_vision_run:
                with open(catalog_file, "r", encoding="utf-8") as f:
                    existing_catalog = json.load(f)
                catalog_Map = {item["file"].replace("\\", "/"): item for item in existing_catalog}

                # A) Pokud uživatel přidal úplně novou fotku, kterou ještě Vision neviděl -> spustíme analýzu
                if any(img not in catalog_Map for img in image_files):
                    print("\n [FÁZE 1B/4] Detekovány nové fotografie v 'img/', spouštím aktualizaci katalogu...")
                    need_vision_run = True
                # B) Pokud uživatel některé fotky smazal (nebo se změnilo pořadí) -> upravíme katalog bez volání API!
                elif list(catalog_Map.keys()) != image_files:
                    print("\n [FÁZE 1B/4] Detekováno smazání/přeskládání fotek v 'img/'! Synchronizuji katalog a přečíslovávám...")
                    synced_catalog = []
                    for new_idx, img_path in enumerate(image_files, start=1):
                        item = catalog_Map[img_path]
                        item["obr_cislo"] = new_idx
                        synced_catalog.append(item)
                    with open(catalog_file, "w", encoding="utf-8") as f:
                        json.dump(synced_catalog, f, ensure_ascii=False, indent=2)
                    # Vynutíme přegenerování textu protokolu (Fáze 2 a 3), protože se změnil počet/číslování fotek!
                    ran_previous_stage = True

            if need_vision_run:
                print("\n [FÁZE 1B/4] Spouštím VisionCuratorAgent (analýza a formátování fotografií)...")
                forbidden_words = self.task_config.get("forbidden_words", [])
                curator = VisionCuratorAgent(model_name=self.models.vision_model,forbidden_words=forbidden_words)
                curator.execute(image_files, verified_sections, task_rules)
                ran_previous_stage = True
            else:
                print("\n [FÁZE 1B/4] Přeskakuji Vision API (katalog 'processed/images_catalog.json' je aktuální).")

            with open(catalog_file, "r", encoding="utf-8") as f:
                img_catalog = json.load(f)
        else:
            img_catalog = []
        # --- MEZIKROK: PYTHON VERIFIKACE MATERIÁLŮ (Ochrana proti halucinacím) ---
        has_numeric_data = bool(valid_files)
        verified_plan = MaterialVerifier.verify(plan_data, img_catalog, has_numeric_data)

        # --- FÁZE 2: NÁVRH PROTOKOLU ---
        if not draft_file.exists() or ran_previous_stage or force_stage in [1, 2]:
            print("\n [FÁZE 2/4] Spouštím QuartoWriterAgent...")
            writer = QuartoWriterAgent(self.models.writer_model, global_rules, task_rules,forbidden_words=forbidden_words)
            writer.execute(verified_plan, computed_summary, generated_plots, draft_file)
            ran_previous_stage = True
        else:
            print("\n [FÁZE 2/4] Přeskakuji (nalezen 'draft_protokol.qmd').")

        with open(draft_file, "r", encoding="utf-8") as f:
            current_qmd = f.read()

       # --- FÁZE 3: INSPEKCE A SCOREBOARD SMYČKA ---
        if not final_file.exists() or ran_previous_stage or force_stage in [1, 2, 3]:
            print("\n [FÁZE 3/4] Spouštím kontrolu kvality (QualityControl + ProtocolInspectorAgent)...")
            inspector = ProtocolInspectorAgent(self.models.inspector_model, global_rules)
            qc = QualityControl(forbidden_words=self.task_config.get("forbidden_words", []),custom_check_fn=self.custom_scoreboard_checks)
            current_qmd = qc.inspect_and_refine(current_qmd, inspector, verified_plan, computed_summary, annotate_fn=self.annotate_and_log_issues)
            final_file.write_text(self.build_yaml_header() + current_qmd, encoding="utf-8")
        else:
            print("\n [FÁZE 3/4] Přeskakuji (nalezen hotový 'protokol.qmd').")

        # --- FÁZE 4: KOMPILACE DO PDF ---
        print("\n [FÁZE 4/4] Spouštím překlad v Quartu...")
        self.compile_pdf()

    #def _clean_fences(self, text: str) -> str:
    #    """Odstraní vnější markdown blok POUZE tehdy, pokud jím agent obalil celý výstup."""
    #    text = text.strip()
    #   for prefix in ["```markdown", "```quarto", "```qmd"]:
    #        if text.startswith(prefix):
    #            text = text[len(prefix):].strip()
    #            if text.endswith("```"):
    #                text = text[:-3].strip()
    #            return text
    #    return text

    def annotate_and_log_issues(self, qmd_body: str, penalties: list[str]) -> str:
            """Pokud zůstaly v kódu prohřešky, najde přesné řádky, vypíše je a označí v .qmd."""
            if not penalties:
                return qmd_body
    
            lines = qmd_body.splitlines()
            report_lines = ["\n" + "="*60, " DIAGNOSTICKÝ AUDIT NEVYŘEŠENÝCH PROHŘEŠKŮ (K RUČNÍ KONTROLE):", "="*60]
    
            for p in penalties:
                report_lines.append(f"• {p}")
                # Pokusíme se najít číslo řádku podle typu chyby
                for idx, line in enumerate(lines, start=1):
                    if ("seaborn" in p and "seaborn" in line) or \
                       ("savefig" in p and ("savefig" in line or "close" in line)) or \
                       ("LaTeX závorky" in p and re.search(r"\\[a-zA-Z]+\{[^{}]+\}", line)) or \
                       ("výpustky" in p and ("..." in line or "DOPLŇ" in line)):
                        report_lines.append(f"    ↳ Nalezeno na řádku {idx}: {line.strip()}")
                        lines[idx - 1] = line + "  # <-- ZDE ZKONTROLOVAT (SCOREBOARD)"
    
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
            print(" ÚSPĚCH! Otevírám protokol.pdf...")
            os.startfile("protokol.pdf")
        else:
            print(" CHYBA KOMPILACE:\n", res.stderr)