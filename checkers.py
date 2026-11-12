import re
import json
import os
import textwrap
from pathlib import Path

class MaterialVerifier:
    """
    FÁZE 2: Čistý Python verifikátor vstupních materiálů (žádné LLM).
    Porovná vyextrahovaný plán ze zadání s reálně dostupnými fotkami a daty.
    """

    @staticmethod
    def verify(plan_data: dict, img_catalog: list[dict], has_numeric_data: bool) -> dict:
        def _norm(text: str) -> str:
            return (str(text).lower().replace("\u2011", "-").replace("–", "-").replace("—", "-"))
        verified_steps = []
        used_images = set()

        for step in plan_data.get("pozadovane_kroky", []):
            norm_sekce = _norm(step.get("nazev_sekce", ""))
            keywords = [_norm(kw) for kw in step.get("klicova_slova", [])]

            matching_images = [
                img for img in img_catalog
                if img["file"].replace("\\", "/") not in used_images
                and (
                    norm_sekce in _norm(img.get("sekce", ""))
                    or any(kw in _norm(img.get("caption", "")) for kw in keywords)
                )
            ]

            if matching_images or has_numeric_data:
                prirazene = [img["file"].replace("\\", "/") for img in matching_images]
                # Přibalíme ke kroku rovnou celé slovníky fotek (s popisky a postupem), aby měl Writer vše na jednom místě!
                step["prirazene_fotky_detail"] = matching_images
                step["prirazene_fotky"] = prirazene
                used_images.update(prirazene)
                verified_steps.append(step)
            else:
                print(f" [MaterialVerifier] Chybí materiály k bodu '{step['nazev_sekce']}'! Přeskakuji tuto sekci.")

        # Pojistka: Pokud zbyly v katalogu fotky, které se nechytly na klíčová slova,
        # přiřadíme je do poslední schválené sekce, aby v protokolu žádná fotka nechyběla.
        unassigned_imgs = [img for img in img_catalog if img["file"].replace("\\", "/") not in used_images]
        if unassigned_imgs and verified_steps:
            verified_steps[-1]["prirazene_fotky_detail"].extend(unassigned_imgs)
            verified_steps[-1]["prirazene_fotky"].extend([img["file"].replace("\\", "/") for img in unassigned_imgs])

        return {
            "teorie_shrnuti": plan_data.get("teorie_shrnuti", ""),
            "schvalene_kroky": verified_steps,
        }


class QualityControl:
    """
    FÁZE 3: Sanitizace výstupu, Scoreboard a samoopravná inspekční smyčka.
    """

    def __init__(self, forbidden_words: list[str] = None, custom_check_fn=None):
        self.forbidden_words = forbidden_words or []
        self.custom_check_fn = custom_check_fn

    def sanitize_output(self, text: str, verified_plan: dict = None) -> str:
        """
        Deterministicky opraví syntaktické chyby Quarto/LaTeX dokumentu,
        sjednotí formátování obrázků a ochrání překlad před pádem pdflatexu.
        """
        # 0. Oříznutí cizí YAML hlavičky nebo úvodního povídání PŘED '# Teoretický úvod'
        if "# Teoretický úvod" in text and "before-body.tex" not in text[:600]:
            text = text[text.index("# Teoretický úvod"):]

        # 1. Automatická oprava ```python na spustitelný Quarto blok ```{python}
        text = re.sub(r"```python\b", "```{python}", text)

        # 2. Automatická oprava cesty k summary.json
        text = text.replace("data/summary.json", "processed/summary.json")

        # 3. Oprava případného poškození Python indexování (pokud agent přepsal [ ] na $$)
        text = re.sub(r"df\$\$(.*?)\$\$", r"df[\1]", text)
        text = re.sub(r"\.iloc\$\$(\d+)\$\$", r".iloc[\1]", text)
        text = re.sub(r"(task\d+)\$\$'(.*?)'\$\$", r"\1['\2']", text)

        # 4. Automatická oprava LaTeX závorek \(...\) a \[...\] (ignoruje odřádkování \\[6pt])
        text = re.sub(r"(?<!\\)\\\[", "$$", text)
        text = re.sub(r"(?<!\\)\\\]", "$$", text)
        text = re.sub(r"(?<!\\)\\\(", "$", text)
        text = re.sub(r"(?<!\\)\\\)", "$", text)

        # 5. Očištění problematických Unicode znaků (nedělitelné spojovníky a úzké mezery)
        text = (
            text.replace("\u2011", "-")
            .replace("\\u2011", "-")
            .replace("\u202f", " ")
            .replace("\\u202f", " ")
        )

        # 6. Oprava Python bloků (zdvojení LaTeX závorek + spojení rozbitých jednořádkových f-stringů)
        def fix_python_block(match):
            block = match.group(0)
            # A) Zdvojení závorek přímo za LaTeX příkazy: \text{low} -> \text{{low}}
            block = re.sub(r"(\\[a-zA-Z]+)\{(?!\{)([^{}]*?)(?<!\})\}", r"\1{{\2}}", block)
            # B) Zdvojení závorek u indexů obsahujících LaTeX příkaz: _{\mathrm{{S}}} -> _{{\mathrm{{S}}}}
            block = re.sub(
                r"([_^])\{(?!\{)([^{}]*?(?:\{\{[^{}]*?\}\}[^{}]*?)*\\[a-zA-Z]+[^{}]*?)(?<!\})\}",
                r"\1{{\2}}",
                block,
            )
            # C) Oprava jednořádkových uvozovek roztažených přes Enter uvnitř display(Markdown(...))
            block = re.sub(
                r'display\(\s*Markdown\(\s*(r?f?)"(?!"")([^"\n]*)\n+\s*([^"]*)"\s*\)\s*\)',
                r'display(Markdown(\1"\2 \3"))',
                block,
            )
            return block

        text = re.sub(r"```\{python\}.*?```", fix_python_block, text, flags=re.DOTALL)

        # Odstranění ručního číslování v nadpisech (např. '# 1. Teoretický úvod' -> '# Teoretický úvod')
        text = re.sub(r"^(#{1,3})\s*\d+(?:\.\d+)*\.?\s+", r"\1 ", text, flags=re.MULTILINE)

        # Pokud nejde o numerickou úlohu, odstraníme případné halucinované Python bloky čtoucí summary.json
        if not os.path.exists("processed/summary.json"):
            text = re.sub(r"```\{python\}[^`]*?summary\.json[^`]*?```", "", text, flags=re.DOTALL)

        # 7. SJEDNOCENÍ VŠECH OBRÁZKŮ A GRAFŮ (Garantované prázdné řádky okolo + napojení na katalog fotek)
        caption_map = {}
        # Nejprve zkusíme vytáhnout popisky přímo z verified_plan v paměti
        if verified_plan:
            for step in verified_plan.get("schvalene_kroky", []):
                for img_item in step.get("prirazene_fotky_detail", []):
                    caption_map[img_item["file"].replace("\\", "/")] = img_item.get("caption", "")

        # Záloha: pokud není verified_plan k dispozici, načteme processed/images_catalog.json
        if not caption_map:
            catalog_path = Path("processed/images_catalog.json")
            if catalog_path.exists():
                try:
                    with open(catalog_path, "r", encoding="utf-8") as cf:
                        img_catalog = json.load(cf)
                    caption_map = {item["file"].replace("\\", "/"): item["caption"] for item in img_catalog}
                except Exception as e:
                    print(f" [Sanitize] Chyba při čtení katalogu obrázků: {e}")

        def format_all_images(match):
            alt_text = match.group(1).strip()
            img_path = match.group(2).strip().replace("\\", "/")
            existing_attrs = match.group(3) or ""

            # POJISTKA: Pokud soubor obrázku na disku fyzicky neexistuje, odstraníme ho z .qmd
            if not os.path.exists(img_path):
                print(f" [Sanitize] Odstraňuji odkaz na neexistující obrázek '{img_path}'.")
                return ""

            # Pokud je obrázek v katalogu fotek (img/...)
            if img_path in caption_map and caption_map[img_path]:
                alt_text = caption_map[img_path]
                attrs = '{height=32% fig-align="center"}'
            else:
                alt_text = re.sub(r"^(?:Obr\.|Obrázek)\s*\d+[\.:]?\s*", "", alt_text, flags=re.IGNORECASE)
                attrs = existing_attrs if existing_attrs else '{width=72% fig-align="center"}'

            # Odstraníme případný název souboru v závorce na konci popisku, např. '(img/34.jpg)'
            alt_text = re.sub(r"\s*\(img/[^)]+\)", "", alt_text)
            return f"\n\n![{alt_text}]({img_path}){attrs}\n\n"

        text = re.sub(r"\!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?", format_all_images, text)

        # Odstranění případných vícenásobných prázdných řádků (max 1 prázdný řádek mezi odstavci)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Zalomení dlouhých řádků v odstavcích pro snadnou ruční editaci .qmd souboru
        text = self._wrap_prose_lines(text, width=95)
        return text.strip() + "\n"

    def evaluate_qmd_scoreboard(self, qmd_text: str, verified_plan: dict) -> tuple[int, list[str]]:
        """
        Vyhodnotí kvalitu .qmd dokumentu na stupnici 0-100 bodů.
        """
        score = 100
        penalties = []

        # 1. Kontrola značek [NEJISTOTA: ...]
        uncertainties = re.findall(r"\[NEJISTOTA:.*?\]", qmd_text)
        if uncertainties:
            score -= 30
            penalties.append(
                f"-30 bodů: V textu zůstaly nevyřešené značky {uncertainties}! "
                f"Ověř je z dat nebo je nahraď boxem '> ** K RUČNÍMU DOPLNĚNÍ:** ...'!"
            )

        # 2. Kontrola cest a Quarto Python bloků
        if "data/summary.json" in qmd_text:
            score -= 50
            penalties.append("-50 bodů: Soubor summary.json leží v 'processed/summary.json', nikoliv v 'data/summary.json'!")

        if "display(df)" in qmd_text:
            score -= 20
            penalties.append("-20 bodů: Zakázané display(df). Použij display(Markdown(df.to_markdown(index=False))).")

        if "Markdown(" in qmd_text and "display(Markdown" not in qmd_text:
            score -= 30
            penalties.append("-30 bodů: Použil jsi Markdown() bez obalení do display().")

        # 3. Kontrola f-stringů mimo Python bloky a nedodělků
        text_outside_code = re.sub(r"```\{python\}.*?```", "", qmd_text, flags=re.DOTALL)
        if re.search(r"\{[a-zA-Z0-9_]+:[.0-9]+f\}", text_outside_code):
            score -= 30
            penalties.append("-30 bodů: Mimo Python bloky f-stringy s formátováním '{promenna:.3f}' nefungují!")

        if "..." in qmd_text or "DOPLŇ" in qmd_text:
            score -= 30
            penalties.append("-30 bodů: V textu zůstaly výpustky '...' nebo 'DOPLŇ'.")

        # 4. Kontrola sekce Závěr (zákaz odrážek)
        if "Závěr" in qmd_text:
            zaver_text = qmd_text.split("Závěr")[-1]
            if re.search(r"^\s*(?:[\*\-]|\d+\.)\s+", zaver_text, re.MULTILINE):
                score -= 40
                penalties.append("-40 bodů: Sekce Závěr nesmí obsahovat odrážky ani číslování, musí jít o souvislý odstavec!")

        # 5. Kontrola nezdvojených LaTeX závorek uvnitř Python f-stringů
        python_blocks = re.findall(r"```\{python\}.*?```", qmd_text, flags=re.DOTALL)
        for block in python_blocks:
            if re.search(r"\\[a-zA-Z]+\{(?!\{)|[_^]\{(?!\{)[^{}]*\\[a-zA-Z]+", block):
                score -= 40
                penalties.append(
                    "-40 bodů: CHYBA SYNTAXE! Uvnitř Python f-stringu jsi použil jednoduché složené závorky "
                    "u LaTeXu. I vnější závorku za podtržítkem MUSÍŠ ZDVOJIT: $I_{{\\mathrm{{S}}}}$!"
                )
                break

        # 6. Kontrola kompletnosti dokumentu a uzavření bloků
        if len(re.findall(r"```", qmd_text)) % 2 != 0:
            score -= 80
            penalties.append("-80 bodů: KRITICKÁ CHYBA! Tvůj výstup je USEKNUTÝ uprostřed bloku (lichý počet ```)!")

        required_sections = ["Teoretický úvod", "Výsledky", "Diskuze", "Závěr"]
        missing_sections = [sec for sec in required_sections if sec.lower() not in qmd_text.lower()]
        if missing_sections:
            score -= 60
            penalties.append(f"-60 bodů: Dokument je NEKOMPLETNÍ! Chybí povinné sekce: {missing_sections}.")

        # 7. Kontrola vložení všech schválených fotografií z verified_plan!
        expected_files = []
        for step in verified_plan.get("schvalene_kroky", []):
            expected_files.extend(step.get("prirazene_fotky", []))

        missing_imgs = [img for img in expected_files if img not in qmd_text]
        if missing_imgs:
            score -= 50
            penalties.append(f"-50 bodů: Do protokolu jsi zapomněl vložit tyto schválené fotografie: {missing_imgs}!")

        # 8. Kontrola zakázaných slov (forbidden_words z task_config.json)
        found_forbidden = [w for w in self.forbidden_words if w.lower() in qmd_text.lower()]
        if found_forbidden:
            score -= 50
            penalties.append(f"-50 bodů: V textu jsi použil ZAKÁZANÁ SLOVA (neprováděné technologie/chyby): {found_forbidden}! Okamžitě je odstraň!")

        # 9. Případné další custom kontroly předané z generátoru
        if self.custom_check_fn:
            c_deduct, c_penalties = self.custom_check_fn(qmd_text)
            score -= c_deduct
            penalties.extend(c_penalties)

        return score, penalties

    def inspect_and_refine(
        self,
        qmd_body: str,
        inspector_agent,
        verified_plan: dict,
        computed_summary: str,
        annotate_fn=None,
    ) -> str:
        """
        Samoopravná while smyčka se sledováním progrese a detekcí zacyklení.
        """
        best_score = -999
        best_qmd = qmd_body
        iteration = 1
        max_iterations = 4
        penalty_history_count = {}

        plan_str = json.dumps(verified_plan, ensure_ascii=False, indent=2)

        while iteration <= max_iterations:
            # Nejprve automaticky opravíme drobnosti přes Regex (zdarma a hned)
            qmd_body = self.sanitize_output(qmd_body, verified_plan)
            score, penalties = self.evaluate_qmd_scoreboard(qmd_body, verified_plan)
            print(f"\n  SCOREBOARD (Iterace {iteration}): {score}/100 bodů")

            if score == 100:
                print("  Dosaženo 100/100 bodů! Návrh je bezchybný.")
                best_qmd = qmd_body
                break

            if score > best_score:
                best_score = score
                best_qmd = qmd_body

            # Detekce zacyklení
            stuck = False
            for p in penalties:
                p_key = p[:45]
                penalty_history_count[p_key] = penalty_history_count.get(p_key, 0) + 1
                if penalty_history_count[p_key] >= 2:
                    print(f"  ZACYKLENÍ DETEKOVÁNO: Chyba '{p_key}...' se opakuje už 2x za sebou!")
                    stuck = True

            if stuck or iteration == max_iterations:
                print(f"  Ukončuji smyčku. Používám nejlepší nalezenou verzi se skóre {best_score}/100 bodů.")
                break

            print("  🔧 Zjištěny prohřešky, předávám k opravě ProtocolInspectorAgentovi:")
            for p in penalties:
                print("    ", p)

            # Zavoláme Inspectora se schváleným plánem místo surového zadání
            candidate_body = inspector_agent.execute(qmd_body, penalties, plan_str, computed_summary)

            if len(candidate_body) > 200:
                qmd_body = candidate_body
            else:
                print(" Inspector vrátil prázdnou/krátkou odpověď, ponechávám předchozí verzi.")

            iteration += 1

        _, final_penalties = self.evaluate_qmd_scoreboard(best_qmd, verified_plan)
        if annotate_fn:
            best_qmd = annotate_fn(best_qmd, final_penalties)
        return self.sanitize_output(best_qmd)
    
    def _wrap_prose_lines(self, text: str, width: int = 95) -> str:
        """
        Zalomí dlouhé odstavce běžného textu na čitelnou šířku (cca 95 znaků),
        aniž by poškodil nadpisy, obrázky, tabulky, Python bloky nebo blokové rovnice.
        """
        wrapped_lines = []
        in_code_block = False
        in_math_block = False

        for line in text.splitlines():
            stripped = line.strip()

            # Sledování spustitelných bloků ```{python} ... ```
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                wrapped_lines.append(line)
                continue

            # Sledování blokové matematiky $$ ... $$
            if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 2:
                wrapped_lines.append(line)
                continue
            if stripped == "$$":
                in_math_block = not in_math_block
                wrapped_lines.append(line)
                continue

            # Pokud jsme uvnitř kódu/matematiky nebo jde o speciální Markdown prvek -> nezalamovat
            is_special_md = (
                not stripped
                or stripped.startswith(("#", "![", "|", ">", "---"))
                or re.match(r"^(?:[\*\-]|\d+\.)\s+", stripped)
            )

            if in_code_block or in_math_block or is_special_md or len(line) <= width:
                wrapped_lines.append(line)
            else:
                # Zalomí běžný odstavec pouze v mezerách mezi slovy (neroztrhne URL ani vzorce bez mezer)
                wrapped_lines.append(
                    textwrap.fill(
                        stripped,
                        width=width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )

        return "\n".join(wrapped_lines)