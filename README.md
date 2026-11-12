# Generator protokolu s vyuzitim LLM pro studenty VUT FEKT

Pokus o multi-agentní systém v Pythonu pro automatizované zpracování laboratorních dat, analýzu fotodokumentace a generování kompletních laboratorních protokolů v typografickém systému **Quarto (`.qmd` $\rightarrow$ LaTeX $\rightarrow$ `.pdf`)**.

Architektura využívá model **Plan-and-Execute** s deterministickou verifikací v Pythonu (`MaterialVerifier` a `QualityControl` se Scoreboardem) a univerzální rozhraní **LiteLLM**, které umožňuje kombinovat modely od **Google Gemini, Groq, OpenAI a Anthropic**.
**Momentálně jsou otestované pouze Groq a Gemini!**
---

## Systémové požadavky

Před instalací Python balíčků musí být v systému nainstalovány tyto nástroje:

1. **Python 3.10 – 3.12** ([python.org](https://www.python.org/))
2. **Quarto CLI** ([quarto.org/docs/get-started](https://quarto.org/docs/get-started/)) – musí být dostupné v systémové `PATH` (ověření v terminálu: `quarto --version`).
3. **LaTeX distribuce pro překlad do PDF:**
   * Nejjednodušší způsob je nechat Quarto nainstalovat si **TinyTeX** jedním příkazem v terminálu:
     ```bash
     quarto install tinytex
     ```
---

## Rychlá instalace (Windows CMD, PowerShell, Linux / macOS)

### 1. Naklonování repozitáře
```bash
git clone <URL_TVEHO_REPOZITARE>
cd protocol_generator
```

### 2. Vytvoření a aktivace virtuálního prostředí (`.venv`)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Pokud PowerShell hlásí chybu oprávnění skriptů, spusť nejprve: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS (Bash / Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalace Python dependenci
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*(Poznámka: Při úplně prvním spuštění s fotografiemi si knihovna `EasyOCR` jednorázově stáhne ~100 MB váhy OCR modelu).*

### 4. Nastavení `.env` souboru
Zkopíruj šablonu `.env.example` do nového souboru `.env` a vyplň svůj API klíč (např. z [Google AI Studio](https://aistudio.google.com/) nebo [Groq Console](https://console.groq.com/)) a své jméno:

```bash
# Windows CMD:
copy .env.example .env

# PowerShell / Linux / macOS:
cp .env.example .env
```

---

## Postup generování protokolu

### Krok 1: Inicializace nové úlohy
Spusť průvodce v kořenové složce a vyplň (kód předmětu, název úlohy, číslo úlohy, datum):
```bash
python init_task.py
```
Skript automaticky vytvoří strukturu složky v `ulohy/<KOD>_uloha<X>_<NAZEV>/` včetně konfiguračního souboru `task_config.json` a RUN `run_task.py`.

### Krok 2: Vložení vstupních souborů do složky úlohy
Do nově vytvořené složky úlohy nahraj podklady:
1. **`zadani.pdf`** – PDF se zadáním laboratorní úlohy (případně celé skriptum -> !pokud te netrapi limit tokenu!; rozsah stran této úlohy nastavíš v `task_config.json` pomocí `"start_page"` a `"end_page"`).
2. **`data/`** – Naměřené tabulky (`.xlsx`, `.xls`, `.csv`), pokud jde o měřicí/numerickou úlohu.
3. **`img/`** – Fotografie z laboratoře (`.jpg`, `.jpeg`, `.png`), pokud jde o praktickou/technologickou úlohu.

### Krok 3: Úprava `task_config.json`
Otevři `ulohy/<tvoje_uloha>/task_config.json` a zkontroluj:
* **`start_page` / `end_page`:** Od které do které strany v `zadani.pdf` se nachází text této úlohy (kritické u vícestránkových skript!).
* **`task_specific_rules`:** Specifické pokyny pro výpočet nebo strukturu (např. *„Vykresli V-A charakteristiku v logaritmickém měřítku“*).
* **`forbidden_words`:** Seznam slov a technologií ze skript, které se v laboratoři **nedělaly** a nesmí se v protokolu objevit (např. `["laser", "flip-chip", "SMD"]`).

### Krok 4: Spuštění generátoru
```bash
python ulohy/<tvoje_uloha>/run_task.py
```

### Krok 5: Ruční úprava a překompilování (Volitelné)
Výsledný zdrojový kód protokolu najdeš v `ulohy/<tvoje_uloha>/protokol.qmd` (text je automaticky zalomený na 95 znaků na řádek pro pohodlnou editaci). Po ruční úpravě textu můžeš PDF přegenerovat okamžitě bez volání AI (0 tokenů):
```bash
cd ulohy/<tvoje_uloha>
quarto render protokol.qmd
```

---

## Omezení a pravidla pro vstupy

Aby generátor fungoval spolehlivě a nespadl při zpracování dat či kompilaci LaTeXu, dodržuj následující pravidla:

### 1. Požadavky na vstupy
* **Použitelné zadání (na FEKTu docela problém) `zadani.pdf`:** PDF musí obsahovat skutečnou textovou vrstvu (označitelný text), nikoliv jen naskenované obrázky stránek bez OCR.
* **Struktura souborů v `data/`:** Podporovány jsou formáty `.xlsx`, `.xls` a `.csv`. Nepoužívej zaheslované sešity ani makra (`.xlsm`). Pokud Excel obsahuje více listů nebo složité vícenásobné tabulky vedle sebe, popiš v `task_specific_rules`, které sloupce/listy má `DataAnalystAgent` použít.
* **Fotografie v `img/`:**
  * Podporované přípony: `.jpg`, `.jpeg`, `.png`
  * Názvy souborů by měly začínat číslem pro správné výchozí řazení (např. `01.jpg`, `02.jpg`) pokud ponecháš default (např. `20261103_093110.jpg`) nástroj rozhodne sám o jejich pořadí
  * Soubor `FEKT_logo.jpg` je rezervován pro hlavičku a je automaticky vynechán z obsahové analýzy.
* **Výběr modelu pro `VISION_MODEL`:** Pokud ručně nastavuješ `VISION_MODEL` v `.env`, **musí jít o multimodální model** podporující vstup obrazu. Čistě textové modely (např. `gpt-oss-120b`) vyhodí chybu `400 BadRequestError`.

### 2. Limity nástroje
* **Ukladani mezivýsledků (`processed/`):**
  * Aby nástroj neplýtval tokeny při opakovaném debugu textu, ukládá si výsledky Fáze 0 (`task_plan.json`), Fáze 1A (`summary.json`) a Fáze 1B (`images_catalog.json`) do složky `processed/`.
  * **Důležité:** Pokud změníš `forbidden_words` nebo `task_specific_rules` a chceš přegenerovat i plán či katalog fotek, **smaž složku `processed/` a soubor `draft_protokol.qmd`** (nebo v `run_task.py` zavolej `generator.run(force_stage=1)`).
* **Detekce smazání/přidání fotek:** Pokud ze složky `img/` pouze smažeš špatnou fotku, generátor při dalším běhu katalog automaticky přečísluje **bez volání VisionCuratorAgenta**. Zavolá se znovu pouze tehdy, když do `img/` přidáš novou fotografii.
* **API Rate Limits (TPM / RPM):** U bezplatných účtů (zejména Groq Free Tier) může při úloze s >30 fotografiemi dojít k vyčerpání denního/minutového limitu tokenů. Pro velké fotografické úlohy důrazně doporučujeme použít **Google Gemini (`gemini/gemini-2.5-flash`)**, který spotřebuje ~5× méně image tokenů.
* **Kontrola kvality a `audit_protokolu.txt`:** Smyčka `QualityControl` provádí maximálně 4 opravné iterace. Pokud se některá chyba zacyklí (zopakuje se 2× po sobě), smyčka se zastaví, použije nejvýše bodovanou verzi a zbývající upozornění vypíše do terminálu a souboru `audit_protokolu.txt` k ruční kontrole.

## Klíčové vlastnosti

* **Multi-Model & Multi-Provider (`LiteLLM`):** Automatická detekce dostupných API klíčů v `.env` a realtime validace modelů. Možnost přiřadit různým rolím (Planner, Vision, Data Analyst, Writer, Inspector) specifické modely.
* **Architektura Plan-and-Execute (`TaskPlanner` + `MaterialVerifier`):** Systém nejprve vyextrahuje požadované kroky ze zadání a v čistém Pythonu je porovná s dodanými fotografiemi a daty. Kroky bez důkazů (např. alternativní technologie zmíněné pouze v teorii) jsou automaticky odfiltrovány, což potencialne eliminuje halucinace.
* **Hybridní Vision + OCR pipeline (`VisionCurator` + `EasyOCR`):** Automatická komprese snímků (úspora >70 % tokenů), oprava EXIF rotace a lokální čtení štítků přístrojů/displejů pomocí `EasyOCR` před samotným voláním multimodálního LLM.
* **Samoopravný výpočetní agent (`DataAnalyst`):** Generuje a v izolovaném subprocesu spouští skript `process_data.py` (`pandas`, `numpy`, `matplotlib`). Obsahuje automatickou kontrolu a až 3 opravné pokusy podle chybového výpisu (`stderr`).
* **Deterministická kontrola a Scoreboard (`QualityControl`):** Automatická oprava LaTeX syntaxe, sjednocení měřítek obrázků, kontrola zakázaných slov (`forbidden_words`), zalamování odstavců na 95 znaků pro lepsi manualni editaci `.qmd` a iterativní revize přes `ProtocolInspectorAgenta`.

---