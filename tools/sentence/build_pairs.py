# -*- coding: utf-8 -*-
"""Генератор пар «последовательность распознанных слов -> предложение».

Запуск::

    python build_pairs.py --count 150000

На выходе в папке ``data`` появятся:

* ``train.tsv`` / ``val.tsv`` / ``test.tsv`` — пары ``вход<TAB>выход``;
  вход — метки через ``|``, выход — токены через пробел;
* ``vocab.txt`` — общий словарь входа и выхода (по одному токену на строку,
  id = номер строки, первые четыре токена — служебные).

Идея: вход — только 100 слов, которые умеет распознавать приложение.
Выход — короткое предложение из форм этих же слов плюс закрытый набор
служебных слов (предлоги, союзы, «есть», «живёт» и т.п.).  Никаких
посторонних содержательных слов модель не придумывает.
"""

import argparse
import itertools
import random
import re
import sys
from pathlib import Path

import pymorphy3

import lexicon as L

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# Тело, которое имеет смысл только в специальных шаблонах
# («паук плетёт паучью сеть», «пчела делает соты»).
SPECIAL_BODY = {"паучья сеть", "соты"}

GENERIC_BODY = [b for b in L.BODY_PARTS if b not in SPECIAL_BODY]

MANUAL_FORMS = {
    "есть": ["есть", "ест", "едят", "ел", "ела", "ело", "ели"],
    "жить": ["жить", "живёт", "живут", "жил", "жила", "жило", "жили"],
    "делать": ["делать", "делает", "делают", "делал", "делала", "делало", "делали"],
    "плести": ["плести", "плетёт", "плетут", "плёл", "плела", "плело", "плели"],
    "летать": ["летать", "летает", "летают", "летал", "летала", "летало", "летали"],
    "плавать": ["плавать", "плавает", "плавают", "плавал", "плавала", "плавало", "плавали"],
    "бегать": ["бегать", "бегает", "бегают", "бегал", "бегала", "бегало", "бегали"],
    "выть": ["выть", "воет", "воют", "выл", "выла", "выло", "выли"],
    "гавкать": ["гавкать", "гавкает", "гавкают", "гавкал", "гавкала", "гавкало", "гавкали"],
    "мигрировать": [
        "мигрировать", "мигрирует", "мигрируют",
        "мигрировал", "мигрировала", "мигрировало", "мигрировали",
    ],
    "ползать": ["ползать", "ползает", "ползают", "ползал", "ползала", "ползало", "ползали"],
    "клевать": ["клевать", "клюёт", "клюют", "клевал", "клевала", "клевало", "клевали"],
    "рычать": ["рычать", "рычит", "рычат", "рычал", "рычала", "рычало", "рычали"],
    "вилять": ["вилять", "виляет", "виляют", "вилял", "виляла", "виляло", "виляли"],
    "больше": ["больше"],
    "меньше": ["меньше"],
    "в": ["в", "во"],
    "с": ["с", "со"],
    "о": ["о", "об"],
    "и": ["и"],
    "а": ["а"],
    "но": ["но"],
    "не": ["не"],
    "это": ["это"],
    "очень": ["очень"],
    "тоже": ["тоже"],
    "только": ["только"],
    "прирученный": [
        "прирученный", "прирученная", "прирученное", "прирученные",
        "прирученного", "прирученной", "прирученных",
        "прирученным", "прирученными",
    ],
}

MORPH = pymorphy3.MorphAnalyzer()
WORD_RE = re.compile(r"[а-яё\-]+")


# ---------------------------------------------------------------------------
# Морфология
# ---------------------------------------------------------------------------

def parses(word: str):
    return MORPH.parse(word)


def best_parse(word: str, pos: str | None = None):
    ps = parses(word)
    if not ps:
        return None
    if pos:
        for p in ps:
            if p.tag.POS == pos:
                return p
    return ps[0]


def parse_first(word: str):
    return best_parse(word)


def _norm_yo(text: str) -> str:
    return text.replace("ё", "е")


def animal_parse(animal: str):
    """Надёжный разбор животного: существительное в именительном падеже.

    Без этого pymorphy3 может выбрать омоним: «осел» -> причастие
    «осевший», «белка» -> родительный падеж «белок» и т.п.
    """
    head = animal.split()[-1]
    ps = parses(head)
    if not ps:
        return None
    target = _norm_yo(head)
    candidates = []
    for p in ps:
        if p.tag.POS != "NOUN":
            continue
        if p.tag.case != "nomn" or p.tag.number != "sing":
            continue
        if _norm_yo(p.word) == target or _norm_yo(p.normal_form) == target:
            candidates.append(p)
    if candidates:
        return max(candidates, key=lambda p: p.score)
    for p in ps:
        if p.tag.POS == "NOUN" and p.tag.case == "nomn" and p.tag.number == "sing":
            return p
    for p in ps:
        if p.tag.POS == "NOUN":
            return p
    return ps[0]


def inflect_word(word: str, grams: set, preferred=None) -> str:
    p = preferred if preferred is not None else parse_first(word)
    if p is None:
        return word
    form = p.inflect(set(grams))
    return form.word if form else word


def inflect_phrase(phrase: str, case: str, number: str = "sing", head_parse=None) -> str:
    """Склоняет словосочетание, согласуя прилагательное с головным словом."""
    parts = phrase.split()
    if not parts:
        return phrase
    head = head_parse if head_parse is not None else parse_first(parts[-1])
    grams = {case, number}
    if number == "sing" and head is not None and head.tag.gender in ("masc", "femn", "neut"):
        grams.add(head.tag.gender)
    forms = []
    for i, part in enumerate(parts):
        preferred = head_parse if (i == len(parts) - 1 and head_parse is not None) else None
        forms.append(inflect_word(part, grams, preferred))
    return " ".join(forms)


def animal_gender(animal: str) -> str:
    head = animal_parse(animal)
    gender = head.tag.gender if head is not None else None
    return gender if gender in ("masc", "femn", "neut") else "masc"


def genitive(animal: str) -> str:
    return inflect_phrase(animal, "gent", "sing", head_parse=animal_parse(animal))


def accusative(animal: str) -> str:
    return inflect_phrase(animal, "accs", "sing", head_parse=animal_parse(animal))


def verb_3sg(lemma: str) -> str:
    p = best_parse(lemma, "VERB")
    if p is None:
        return lemma
    form = p.inflect({"VERB", "3per", "sing", "pres"})
    return form.word if form else lemma


def body_form(body: str, positive: bool) -> str:
    number = L.BODY_NUMBER.get(body, "sing")
    case = "nomn" if positive else "gent"
    if " " in body:
        return inflect_phrase(body, case, number)
    return inflect_word(body, {case, number})


def category_plural(category: str) -> str:
    if " " in category:
        return inflect_phrase(category, "nomn", "plur")
    return inflect_word(category, {"nomn", "plur"})


def tokens(*parts) -> list:
    out = []
    for part in parts:
        if part:
            out.extend(str(part).split())
    return out


# ---------------------------------------------------------------------------
# Сборка одного предложения по набору меток
# ---------------------------------------------------------------------------

def action_sentence(animal: str, action: str):
    allowed = L.ACTION_ALLOWED.get(action)
    if allowed is None or animal not in allowed:
        return None

    if action == "вилять":
        return tokens(animal, verb_3sg("вилять"), inflect_word("хвост", {"sing", "ablt"}), ".")
    if action == "делать выводы":
        return tokens(animal, verb_3sg("делать"), inflect_word("вывод", {"plur", "accs"}), ".")

    verb = L.ACTION_NOUNS.get(action, action)
    return tokens(animal, verb_3sg(verb), ".")


def body_sentence(animal: str, body: str):
    if body in SPECIAL_BODY:
        return None
    positive = body in L.body_parts_for(animal)
    form = body_form(body, positive)
    if positive:
        return tokens("у", genitive(animal), "есть", form, ".")
    return tokens("у", genitive(animal), "нет", form, ".")


def place_sentence(animal: str, place: str):
    if place not in L.places_for(animal):
        return None
    return tokens(animal, "живёт", "в", inflect_phrase(place, "loct", "sing"), ".")


def adjective_sentence(animal: str, adjective: str):
    if adjective == L.SPECIAL_ADJECTIVE:
        if animal not in L.DOMESTIC:
            return None
        return tokens(
            animal, "—",
            inflect_word("прирученный", {"nomn", "sing", "neut"}),
            "животное", ".",
        )
    allowed = L.ADJECTIVE_ALLOWED.get(adjective)
    if allowed is not None and animal not in allowed:
        return None
    gender = animal_gender(animal)
    return tokens(animal, inflect_word(adjective, {"nomn", "sing", gender}), ".")


def category_sentence(animal: str, category: str):
    if category == "дикое животное" and animal not in L.WILD:
        return None
    if category == "домашнее животное" and animal not in L.DOMESTIC:
        return None
    if category == "водоплавающее" and animal not in L.WATER_ANIMALS:
        return None
    return tokens(animal, "—", category, ".")


def personality_sentence(animal: str, noun: str):
    if animal not in L.PERSONALITY.get(noun, set()):
        return None
    return tokens(animal, "—", noun, ".")


def two_animals_sentence(a: str, b: str):
    # 1. Хищник и жертва.
    if b in L.PREDATOR_PREY.get(a, set()):
        return tokens(a, "ест", accusative(b), ".")
    if a in L.PREDATOR_PREY.get(b, set()):
        return tokens(b, "ест", accusative(a), ".")

    # 2. Заметная разница в размере (минимум два класса).
    ra, rb = L.SIZE_RANK.get(a), L.SIZE_RANK.get(b)
    if ra is not None and rb is not None and abs(ra - rb) >= 2:
        big, small = (a, b) if ra > rb else (b, a)
        return tokens(big, "больше", genitive(small), ".")

    # 3. Общая категория.
    if a in L.DOMESTIC and b in L.DOMESTIC:
        plural = "домашние животные"
    elif a in L.WATER_ANIMALS and b in L.WATER_ANIMALS:
        plural = "водоплавающие"
    elif a in L.WILD and b in L.WILD:
        plural = "дикие животные"
    else:
        plural = "животные"

    x, y = a, b
    return tokens(x, "и", y, "—", plural, ".")


def two_animals_category(animals: list, category: str):
    if len(animals) != 2:
        return None
    a, b = animals
    if category == "дикое животное" and not (a in L.WILD and b in L.WILD):
        return None
    if category == "домашнее животное" and not (a in L.DOMESTIC and b in L.DOMESTIC):
        return None
    if category == "водоплавающее" and not (a in L.WATER_ANIMALS and b in L.WATER_ANIMALS):
        return None
    x, y = a, b
    return tokens(x, "и", y, "—", category_plural(category), ".")


def list_sentence(animals: list):
    # Порядок слов сохраняем таким, как его распознало приложение.
    animals = list(dict.fromkeys(animals))[:4]
    if len(animals) < 3:
        return None
    out = []
    for i, animal in enumerate(animals):
        if i:
            out.append("," if i < len(animals) - 1 else "и")
        out.append(animal)
    out.append(".")
    return out


def resolve(labels) -> list | None:
    """Каноническое предложение для набора меток (порядок не важен)."""
    labels = list(dict.fromkeys(labels))
    if not labels:
        return None

    animals = [x for x in labels if x in L.ANIMALS]
    bodies = [x for x in labels if x in L.BODY_PARTS]
    places = [x for x in labels if x in L.PLACES_ALL]
    adjectives = [x for x in labels if x in L.ADJECTIVES or x == L.SPECIAL_ADJECTIVE]
    actions = [x for x in labels if x in L.ACTION_VERBS or x in L.ACTION_NOUNS]
    categories = [x for x in labels if x in L.CATEGORIES]
    personalities = [x for x in labels if x in L.PERSONALITY]

    # Одно слово.
    if len(labels) == 1:
        word = labels[0]
        if word in L.ANIMALS or word in L.BODY_PARTS or word in L.PLACES_ALL:
            return tokens(word, ".")
        if word in {"нашествие", "помесь", "воротник", "корм для домашних животных"}:
            return tokens(word, ".")
        return None

    # Специальные шаблоны.
    if set(labels) == {"паук", "паучья сеть"}:
        return tokens("паук", verb_3sg("плести"),
                      inflect_phrase("паучья сеть", "accs", "sing"), ".")
    if set(labels) == {"пчела", "соты"}:
        return tokens("пчела", verb_3sg("делать"),
                      inflect_word("соты", {"plur", "accs"}), ".")

    if len(labels) == 2:
        if len(animals) == 1 and len(actions) == 1:
            return action_sentence(animals[0], actions[0])
        if len(animals) == 1 and len(bodies) == 1:
            return body_sentence(animals[0], bodies[0])
        if len(animals) == 1 and len(places) == 1:
            return place_sentence(animals[0], places[0])
        if len(animals) == 1 and len(adjectives) == 1:
            return adjective_sentence(animals[0], adjectives[0])
        if len(animals) == 1 and len(categories) == 1:
            return category_sentence(animals[0], categories[0])
        if len(animals) == 1 and len(personalities) == 1:
            return personality_sentence(animals[0], personalities[0])
        if len(animals) == 2:
            return two_animals_sentence(animals[0], animals[1])
        return None

    if len(labels) == 3:
        if len(animals) == 2 and len(categories) == 1:
            return two_animals_category(animals, categories[0])
        if len(animals) == 3:
            return list_sentence(animals)
        return None

    if len(labels) == 4 and len(animals) == 4:
        return list_sentence(animals)
    return None


# ---------------------------------------------------------------------------
# Шаблоны-затравки
# ---------------------------------------------------------------------------

def t_single(rng):
    return [rng.choice(L.ANIMALS)]


def t_body(rng):
    return [rng.choice(L.ANIMALS), rng.choice(GENERIC_BODY)]


def t_place(rng):
    return [rng.choice(L.ANIMALS), rng.choice(sorted(L.PLACES_ALL))]


def t_adjective(rng):
    return [rng.choice(L.ANIMALS), rng.choice(L.ADJECTIVES)]


def t_privet(rng):
    return [rng.choice(sorted(L.DOMESTIC)), L.SPECIAL_ADJECTIVE]


def t_category(rng):
    return [rng.choice(L.ANIMALS), rng.choice(L.CATEGORIES)]


def t_two(rng):
    return [rng.choice(L.ANIMALS), rng.choice(L.ANIMALS)]


def t_three(rng):
    return [rng.choice(L.ANIMALS) for _ in range(3)]


def t_four(rng):
    return [rng.choice(L.ANIMALS) for _ in range(4)]


def t_action(rng):
    return [rng.choice(L.ANIMALS), rng.choice(L.ACTION_VERBS)]


def t_action_noun(rng):
    return [rng.choice(L.ANIMALS), rng.choice(sorted(L.ACTION_NOUNS))]


def t_web(rng):
    return ["паук", "паучья сеть"]


def t_honey(rng):
    return ["пчела", "соты"]


def t_tail(rng):
    return ["собака", "вилять"]


def t_conclusions(rng):
    return ["обезьяна", "делать выводы"]


def t_personality(rng):
    noun = rng.choice(sorted(L.PERSONALITY))
    return [rng.choice(sorted(L.PERSONALITY[noun])), noun]


def t_two_category(rng):
    return [rng.choice(L.ANIMALS), rng.choice(L.ANIMALS), rng.choice(L.CATEGORIES)]


TEMPLATES = [
    (t_single, 4),
    (t_body, 12),
    (t_place, 8),
    (t_adjective, 8),
    (t_privet, 3),
    (t_category, 8),
    (t_two, 10),
    (t_three, 6),
    (t_four, 4),
    (t_action, 8),
    (t_action_noun, 5),
    (t_web, 1),
    (t_honey, 1),
    (t_tail, 1),
    (t_conclusions, 1),
    (t_personality, 2),
    (t_two_category, 6),
]


def sample_seed(rng) -> list:
    funcs = [f for f, _ in TEMPLATES]
    weights = [w for _, w in TEMPLATES]
    return rng.choices(funcs, weights=weights, k=1)[0](rng)


# ---------------------------------------------------------------------------
# Шум
# ---------------------------------------------------------------------------

def apply_noise(labels: list, rng) -> list:
    labels = list(dict.fromkeys(labels))

    # Иногда теряем одно слово (распознавание не всё услышало).
    if len(labels) > 1 and rng.random() < 0.08:
        labels.pop(rng.randrange(len(labels)))

    # Иногда путаем животное/часть тела с другим словом того же класса.
    if labels and rng.random() < 0.05:
        i = rng.randrange(len(labels))
        if labels[i] in L.ANIMALS:
            labels[i] = rng.choice(L.ANIMALS)
        elif labels[i] in GENERIC_BODY:
            labels[i] = rng.choice(GENERIC_BODY)

    # Иногда слово повторяется.
    if labels and rng.random() < 0.03:
        labels.append(rng.choice(labels))

    return list(dict.fromkeys(labels))


# ---------------------------------------------------------------------------
# Словарь
# ---------------------------------------------------------------------------

def add_lexeme(token: str, vocab: set, pos: str | None = None) -> None:
    p = best_parse(token, pos)
    if p is None:
        vocab.add(token.lower())
        return
    for form in p.lexeme:
        word = form.word.lower()
        if WORD_RE.fullmatch(word):
            vocab.add(word)
    if WORD_RE.fullmatch(token.lower()):
        vocab.add(token.lower())


def build_vocab(words: list, pairs: list) -> list:
    vocab = set()

    for word in words:
        for token in word.split():
            add_lexeme(token, vocab)

    verb_lemmas = {"есть", "жить", "делать", "плести", "летать", "плавать", "бегать"}
    for lemma in L.FUNCTION_LEMMAS:
        add_lexeme(lemma, vocab, "VERB" if lemma in verb_lemmas else None)

    for lemma in L.OUTPUT_HELPER_LEMMAS:
        add_lexeme(lemma, vocab)

    for forms in MANUAL_FORMS.values():
        vocab.update(forms)

    vocab.update(L.PUNCTUATION)
    vocab.update(words)  # полные метки входа, в том числе «летучая мышь»

    for out in pairs:
        vocab.update(out)

    vocab.discard("")
    return SPECIALS + sorted(vocab - set(SPECIALS))


# ---------------------------------------------------------------------------
# Проверка данных
# ---------------------------------------------------------------------------

def validate_lexicon(words: list) -> None:
    word_set = set(words)

    groups = {
        "ANIMALS": L.ANIMALS,
        "BIRDS": L.BIRDS,
        "INSECTS": L.INSECTS,
        "REPTILES": L.REPTILES,
        "AMPHIBIANS": L.AMPHIBIANS,
        "FISH": L.FISH,
        "DOMESTIC": L.DOMESTIC,
        "WATER_ANIMALS": L.WATER_ANIMALS,
        "HOOFED": L.HOOFED,
        "HORNED": L.HORNED,
        "BODY_PARTS": L.BODY_PARTS,
        "PLACES_ALL": L.PLACES_ALL,
        "CATEGORIES": L.CATEGORIES,
        "ADJECTIVES": L.ADJECTIVES,
        "PERSONALITY": set(L.PERSONALITY),
        "ACTION_NOUNS": set(L.ACTION_NOUNS),
        "UNUSED_WORDS": set(L.UNUSED_WORDS),
        "PREDATOR_PREY keys": set(L.PREDATOR_PREY),
    }
    for name, group in groups.items():
        missing = set(group) - word_set
        if missing:
            raise SystemExit(f"{name}: этих слов нет в words.txt: {sorted(missing)}")

    for predator, prey in L.PREDATOR_PREY.items():
        missing = set(prey) - word_set
        if missing:
            raise SystemExit(f"PREDATOR_PREY[{predator}]: нет в words.txt: {sorted(missing)}")

    missing = set(L.SIZE_RANK) - word_set
    if missing:
        raise SystemExit(f"SIZE_RANK: нет в words.txt: {sorted(missing)}")

    classified = (
        set(L.ANIMALS) | set(L.BODY_PARTS) | set(L.ACTION_VERBS)
        | set(L.ACTION_NOUNS) | set(L.PLACES_ALL) | set(L.CATEGORIES)
        | set(L.ADJECTIVES) | {L.SPECIAL_ADJECTIVE} | set(L.PERSONALITY)
        | set(L.UNUSED_WORDS) | {"населять", "вилять"}
    )
    unclassified = word_set - classified
    if unclassified:
        raise SystemExit(f"Не классифицированы слова: {sorted(unclassified)}")

    extra = classified - word_set
    if extra:
        raise SystemExit(f"В классификации есть лишние слова: {sorted(extra)}")


# ---------------------------------------------------------------------------
# Запись
# ---------------------------------------------------------------------------

def write_tsv(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for inp, out in rows:
            f.write(f"{inp}\t{out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор пар для сборки предложений")
    parser.add_argument("--words", default="words.txt", help="файл со 100 словами")
    parser.add_argument("--out", default="data", help="папка для train/val/test/vocab")
    parser.add_argument("--count", type=int, default=150000, help="сколько уникальных пар хотим")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    words_path = Path(args.words)
    if not words_path.exists():
        raise SystemExit(f"Нет файла {words_path}")

    words = [
        line.strip() for line in words_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(words) != 100:
        raise SystemExit(f"Ожидалось 100 слов, а в {words_path} их {len(words)}")

    validate_lexicon(words)

    rng = random.Random(args.seed)
    by_input: dict = {}
    pairs: dict = {}
    conflicts = 0
    attempts = 0
    max_attempts = max(1000, args.count * 40)

    while len(pairs) < args.count and attempts < max_attempts:
        attempts += 1
        labels = sample_seed(rng)
        labels = apply_noise(labels, rng)
        if not labels:
            continue

        # Один случайный порядок входа за попытку: так модель увидит
        # разные порядки, но датасет не раздувается перестановками.
        perm = tuple(rng.sample(labels, len(labels)))
        out = resolve(perm)
        if not out:
            continue

        key = (perm, tuple(out))
        prev = by_input.get(perm)
        if prev is not None and prev != tuple(out):
            conflicts += 1
            continue
        by_input[perm] = tuple(out)
        pairs[key] = out

    if not pairs:
        raise SystemExit("Не удалось сгенерировать ни одной пары")

    rows = [("|".join(perm), " ".join(out)) for (perm, out) in pairs]
    rng.shuffle(rows)

    n = len(rows)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.05))
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:]

    vocab = build_vocab(words, [out for (_, out) in pairs])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "train.tsv", train)
    write_tsv(out_dir / "val.tsv", val)
    write_tsv(out_dir / "test.tsv", test)
    with (out_dir / "vocab.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(vocab) + "\n")

    print(f"Сгенерировано пар: {len(rows)} (попыток: {attempts}, конфликтов: {conflicts})")
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Словарь: {len(vocab)} токенов (включая {len(SPECIALS)} служебных)")
    print(f"Файлы: {out_dir.resolve()}")
    print()
    print("Примеры:")
    for inp, out in rows[:10]:
        print(f"  {inp.replace('|', ' + '):45s} -> {out}")


if __name__ == "__main__":
    main()
