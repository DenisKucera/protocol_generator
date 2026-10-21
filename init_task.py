import os
import json
import shutil
from pathlib import Path

def init_new_task():
    print("=== INICIALIZACE NOVÉ LABORATORNÍ ÚLOHY ===")
    kod = input("Kód předmětu (např. MPC-MPR): ").strip() or "MPC-MPR"
    predmet = input("Název předmětu (např. Moderní mikroelektronické prvky): ").strip()
    cislo = input("Číslo úlohy (např. 1): ").strip() or "1"
    nazev = input("Název úlohy (např. Měření V-A charakteristik): ").strip()
    datum = input("Datum měření (např. 21. 10. 2026): ").strip()

    # Vytvoření bezpečného názvu složky
    safe_name = f"{kod}_uloha{cislo}_{nazev[:20].replace(' ', '_')}"
    task_dir = Path("ulohy") / safe_name
    
    (task_dir / "data").mkdir(parents=True, exist_ok=True)
    (task_dir / "img").mkdir(parents=True, exist_ok=True)

    # Zkopírování loga FEKT z root/img, pokud existuje
    root_logo = Path("img/FEKT_logo.jpg")
    if root_logo.exists():
        shutil.copy(root_logo, task_dir / "img" / "FEKT_logo.jpg")

    # 1. Vytvoření task_config.json
    task_config = {
        "lab_metadata": {
            "title": nazev,
            "predmet": predmet,
            "kod_predmetu": kod,
            "cislo_ulohy": cislo,
            "author": "Denis Kučera",
            "student_id": "256655",
            "datum_mereni": datum,
            "spolupracoval": "",
            "start_page": 1,
            "end_page": 5
        },
        "task_specific_rules": [
            "Zde dopiš specifické požadavky pro tuto úlohu (např. jaké veličiny jsou na osách X a Y, co se má spočítat)."
        ],
        "forbidden_words": []
    }
    with open(task_dir / "task_config.json", "w", encoding="utf-8") as f:
        json.dump(task_config, f, ensure_ascii=False, indent=2)

    # 2. Vytvoření dědičného spouštěcího skriptu run_task.py přímo ve složce úlohy
    run_task_code = f'''import sys
from pathlib import Path

# Přidání root složky do sys.path, abychom mohli importovat rodičovskou třídu
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from generator_core import BaseProtocolGenerator

class TaskProtocolGenerator(BaseProtocolGenerator):
    """
    Specifická instance generátoru pro úlohu: {nazev} ({kod})
    Dědí veškerou logiku z BaseProtocolGenerator.
    """
    
    def custom_scoreboard_checks(self, qmd_text: str) -> tuple[int, list[str]]:
        # Nejprve zavoláme základní kontrolu zakázaných slov z rodiče
        deduction, penalties = super().custom_scoreboard_checks(qmd_text)
        
        # Zde můžeš snadno přidat Python kontroly specifické JEN pro tuto úlohu:
        # Např.:
        # if "set_xscale('log')" not in qmd_text:
        #     deduction += 30
        #     penalties.append("-30 bodů: Osa X musí být v logaritmickém měřítku!")
            
        return deduction, penalties

if __name__ == "__main__":
    generator = TaskProtocolGenerator(task_dir=Path(__file__).parent)
    generator.run()
'''
    with open(task_dir / "run_task.py", "w", encoding="utf-8") as f:
        f.write(run_task_code)

    print(f"\n Úloha byla úspěšně vytvořena ve složce: {task_dir}")
    print("Další kroky:")
    print(f"  1. Vlož naměřená data (.xlsx / .csv) do '{task_dir}/data/'")
    print(f"  2. Vloz obrazky (.jpg, .png,...) do  '{task_dir}/img/' ")
    print(f"  3. Vlož skripta jako '{task_dir}/zadani.pdf'")
    print(f"  4. Uprav specifická pravidla v '{task_dir}/task_config.json'")
    print(f"  5. Spusť generování příkazem: python {task_dir}/run_task.py")

if __name__ == "__main__":
    init_new_task()