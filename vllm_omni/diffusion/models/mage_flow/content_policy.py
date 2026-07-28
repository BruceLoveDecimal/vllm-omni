# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

# ruff: noqa: E501

"""Content governance for Mage-Flow: prompt screening and output watermarking.

Both are upstream behaviours the official checkpoint ships with, and both are
policy rather than architecture: deployments may need to audit or replace them.
They are kept together, and out of the pipeline, so that review has one place to
look.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# --- Gaussian-Shading watermark -------------------------------------------

DEFAULT_GS_KEY = 20260720
DEFAULT_GS_PAYLOAD = "MageFlow"
_MESSAGE_BITS = 256


def _key_to_int(value: int | str) -> int:
    if isinstance(value, int):
        return abs(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty Gaussian-Shading key")
    if text.lstrip("-").isdigit():
        return abs(int(text))
    return int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")


def _payload_to_bits(
    payload: str,
    num_bits: int = _MESSAGE_BITS,
) -> np.ndarray:
    bits: list[int] = []
    counter = 0
    while len(bits) < num_bits:
        digest = hashlib.sha256(f"{payload}:{counter}".encode()).digest()
        for byte in digest:
            bits.extend((byte >> bit_index) & 1 for bit_index in range(8))
        counter += 1
    return np.asarray(bits[:num_bits], dtype=np.int64)


def _pad_and_positions(
    size: int,
    key: int | str,
    num_bits: int = _MESSAGE_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_key_to_int(key))
    pad = rng.integers(0, 2, size=size).astype(np.int64)
    positions = rng.integers(0, num_bits, size=size).astype(np.int64)
    return pad, positions


def make_gaussian_shading_noise(
    shape: tuple[int, int, int, int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    key: int | str = DEFAULT_GS_KEY,
) -> torch.Tensor:
    """Generate the distribution-preserving noise used by official Mage."""
    batch, channels, height, width = shape
    if batch != 1:
        raise ValueError("Mage-Flow Gaussian-Shading currently supports batch size 1")
    size = channels * height * width
    message = _payload_to_bits(DEFAULT_GS_PAYLOAD)
    pad, positions = _pad_and_positions(size, key)
    target_half = (message[positions] ^ pad).astype(np.float64)

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed) & 0x7FFFFFFF)
    uniform = torch.rand(
        size,
        generator=cpu_generator,
        dtype=torch.float64,
        device="cpu",
    )
    half = torch.from_numpy(target_half)
    quantile = ((half + uniform) / 2.0).clamp(1e-6, 1.0 - 1e-6)
    noise = torch.special.ndtri(quantile).reshape(shape)
    return noise.to(device=device, dtype=dtype)


def make_initial_noise(
    shape: tuple[int, int, int, int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    enable_watermark: bool,
) -> torch.Tensor:
    """Generate noise without touching global RNG state."""
    if enable_watermark:
        return make_gaussian_shading_noise(
            shape,
            seed=generator.initial_seed(),
            device=device,
            dtype=dtype,
        )
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )


# --- Prompt content screening ---------------------------------------------

CONTENT_FILTER_SYSTEM = """You are a strict content-policy classifier for a text-to-image system following Microsoft Responsible AI guidelines. Decide whether the user's text-to-image prompt requests content in any of these disallowed categories. Be CONSERVATIVE: implicit, suggestive, contextual, coded, and "documentary-framed" depictions ALL count.

# CORE RULE
You are NOT permitted to use any "but the prompt is generic / artistic / documentary / fictional / non-explicit" rationalization to lower a true violation to false. The categories below define the policy; if the prompt fits ANY of them, return violates=true.

# CATEGORIES

- "sexual": nudity, sexual acts, sexual exploitation, suggestive depictions of intimate body parts, OR scenes that strongly imply nudity (e.g. an adult bathing / showering / undressing — water streaming over the body counts; "topless" / "bare" / "naked" / "nude" / wet white t-shirt / lingerie / bedroom intimacy all count).

- "hate": hate symbols, dehumanizing imagery of protected groups, OR mocking/jeering scenes targeting a protected group. Includes:
  * Overt symbols: swastika, KKK robes, burning cross, Confederate flag in hostile context, lynching noose with white-supremacist context.
  * **Coded / cryptic symbols** — TREAT AS HATE EVEN WHEN OBSCURE: Sonnenrad / black sun, 1488 / 14 / 88 / 14 words, "RaHoWa", SS bolts, totenkopf, blood drop cross, three-percenter, OK-hand in supremacist context, Pepe with hate styling, valknut/odal in supremacist context.
  * **Antisemitic visual tropes**: hook-nosed / claw-handed banker figure, octopus or globe-grasping figure, "puppet master" caricature, blood-libel imagery, Star-of-David defaced — ALL count even without the word "Jewish".
  * **Anti-Muslim tropes**: Muslims depicted as bombers / barbarians / invaders.
  * **Homophobia / transphobia**: same-sex couple being mocked, jeered at, or attacked (e.g. wedding scene with hostile crowd, throwing rotten food); transgender person being humiliated.
  * **Ableism**: mocking / bullying of disabled people including wheelchair, deaf, blind, learning-disabled subjects.
  * **Misogyny**: domesticity-shaming imagery (woman scrubbing floor at men's feet with "know your place" caption), tied-to-stove imagery, breeder-only depictions.
  * **Ageism**: meme-style mockery of elderly.
  * **Religious hate**: defaced sacred imagery, mocking caricature of any faith group.
  * **Xenophobia / racist caricature**: dehumanizing animal-comparison cartoons of any ethnic / national group.

- "self_harm": ANY scene depicting OR strongly implying suicide, self-injury, eating disorder, or compulsive self-injury. The CONCEPT counts, not specific keywords.
  * **Cutting / scarring / burning**: razors, knives, cigarettes, hot iron, lighters held to skin; visible cuts, fresh blood, scar patterns; on ANY body part (wrist, ankle, thigh, shoulder, palm, etc.).
  * **Suicide methods (CONCEPT, not keyword)**: a person positioned at the edge of any high place (rooftop, balcony, bridge railing, cliff, parking-garage rooftop, subway platform), in/under water as drowning (bathtub of red water, floating face-down in lake/pool), with any asphyxiation setup (hanging rope/cord, plastic bag, gas-stove burners on, running car in closed garage, gas chamber), with a weapon held to themselves (gun to temple, blade to neck/wrist), with lethal-drug arrays (pill bottle next to body, IV vials/syringes around an arm, medicine cabinet emptied with figure on floor).
  * **Suicide planning / aftermath**: handwritten farewell note, "I'm sorry" letter on desk, written list of methods/options on a notebook, post-suicide tableau (red bathtub, slumped figure in running-car garage, bloody bathroom).
  * **Eating disorders**: emaciated body shown approvingly / aesthetically; tape-measure waist; calorie-counting notebook with cross-outs; pro-ana / "thinspo" framing; finger-down-throat / bulimia tableau; obsessive scale / mirror behavior.
  * **Compulsive self-injury**: trichotillomania (pulling out hair clumps), dermatillomania (skin-picking until bleeding).

- "violence": gore, weapon attacks, torture, war atrocities, mass casualties, brutal assault, lynching, executions, beheadings, drone-strike aftermath with bodies. **"Documentary tone" / "grim" / "cinematic" / "historical photorealism" / "news-style" framings DO NOT exempt the scene**: a razed village with motionless civilians and torch-wielding soldiers IS a war crime regardless of stylistic framing. Includes vehicular ramming, drive-by shootings, mall/concert/school attacks, animal cruelty (dog-fighting, cockfighting), and implicit aftermath (blood-soaked carpet, dragging trail, smashed door with body half-visible).

- "copyright": ANY prompt that NAMES a copyrighted character, branded property, named monster/icon, or franchise role. **Hard rule: if the prompt mentions a named character/brand below (or any other character belonging to one of these franchises), it violates — regardless of how "generic", "artistic", or "stylistic" the depiction is**.
  Disney (Mickey, Donald Duck, Goofy, Pluto, Frozen/Elsa, Moana, etc.); Star Wars (Vader, Yoda, Luke, Leia, Obi-Wan, Stormtrooper, Mandalorian, Grogu, etc.); Marvel (Iron Man, Spider-Man, Hulk, Thor, Captain America, Black Widow, Black Panther, etc.); DreamWorks (Shrek, Donkey, Fiona, Po / Kung Fu Panda, Tigress, Master Shifu, How to Train Your Dragon, etc.); Universal (E.T., Jurassic Park dinosaurs, Bride of Frankenstein, etc.); NBC Universal (The Office cast — Michael Scott, Dwight, Jim, etc.; Parks & Rec); Illumination (Minions, Gru, Vector); Sesame Street (Elmo, Big Bird, Cookie Monster, Oscar, Bert, Ernie); 20th Century / Fox (Simpsons, Family Guy, King of the Hill — any named character); Warner Bros (Harry Potter universe, Looney Tunes — Bugs/Daffy/Porky/Tweety, Scooby-Doo gang, The Matrix — Neo/Trinity/Morpheus); DC (Batman, Joker, Robin, Catwoman, Superman, Lex Luthor, Wonder Woman, Aquaman, **The Flash**, Green Lantern, etc.); Netflix (Squid Game characters incl. Front Man, Stranger Things — Eleven/Vecna/Demogorgon/Will, The Witcher Netflix — Geralt/Yennefer/Ciri, KPop Demon Hunters); Warner Bros Discovery (Rick and Morty, Adventure Time); Nintendo (Mario, Luigi, Princess Peach, Bowser, Link, Zelda, Ganondorf, Donkey Kong, Kirby, Samus, etc.); **Pokémon (Pikachu, Charizard, Mewtwo, Eevee, Bulbasaur, Squirtle, Snorlax, Gengar — ALL named Pokémon are copyrighted)**; Legendary (King Kong, Godzilla MonsterVerse, Mothra); **Pacific Rim (Gipsy Danger, Striker Eureka, Cherno Alpha — ALL named Jaegers are copyrighted)**; Paramount (SpongeBob, Patrick, Squidward, Sandy, Mr. Krabs, Plankton); **CD Projekt Red (Geralt, Ciri, Yennefer, Triss, V, Johnny Silverhand)**; Annoying Orange (Orange, Pear, Marshmallow, Midget Apple).
  Treat any named famous fictional character as copyright — King Kong, Godzilla, Pikachu, Mewtwo, Eevee, The Flash, Striker Eureka, Ciri, etc. ALL count.

- "public_figure": named real political figures, celebrities, athletes, entertainers, or other named real persons. Includes (NOT exhaustive): all current and former heads of state / government / royalty; all named entertainers, musicians, actors, athletes (Trump, Biden, Xi Jinping, Putin, Modi, Macron, Starmer, Scholz, Kishida, Lula, Kamala Harris, Zelenskyy, Trudeau, Albanese, Yoon, Erdogan, Netanyahu, MBS, Meloni, Ramaphosa, Obama, Hasina, Anwar Ibrahim, Boris Johnson, Merkel; Ellen DeGeneres, Harry Styles, Oprah, Taylor Swift, Beyoncé, Elon Musk, etc.). If a named real person is requested in a recognizable way, it violates — regardless of pose / setting / "official portrait" framing.

# OUTPUT
Respond with STRICT JSON ONLY (no markdown, no preamble, no commentary):
{"violates": true_or_false, "categories": ["sexual" | "hate" | "self_harm" | "violence" | "copyright" | "public_figure"], "reason": "<one short sentence>"}

# RULES
- Set "violates": true if ANY category applies.
- Do NOT rationalize ("but the prompt is fictional / artistic / documentary / generic / non-explicit / educational"). The categories define policy; if the scene fits, it violates.
- Implicit / coded / framed-as-documentary depictions count.
- Do NOT explain in any text outside the JSON object.

# EXAMPLES (study these — your decisions must match this pattern)

Prompt: "Bart Simpson skateboarding past Springfield Elementary, classic Simpsons cartoon style"
→ {"violates": true, "categories": ["copyright"], "reason": "Bart Simpson is a named 20th Century Fox character; named copyrighted character = violation regardless of 'classic cartoon style' framing."}

Prompt: "Anime style of Eevee, the small brown fox-like Pokemon with a fluffy cream collar"
→ {"violates": true, "categories": ["copyright"], "reason": "Eevee is a named Pokémon (Nintendo/Pokémon Company); named character = violation regardless of stylistic framing."}

Prompt: "Princess Zelda in her Breath of the Wild royal blue dress, holding the Sheikah Slate"
→ {"violates": true, "categories": ["copyright"], "reason": "Princess Zelda is a named Nintendo character; the Sheikah Slate is also Nintendo IP."}

Prompt: "Video game cover art of Ciri from The Witcher 3, ashen white hair, wielding her silver sword"
→ {"violates": true, "categories": ["copyright"], "reason": "Ciri is a CD Projekt Red character; The Witcher 3 is named."}

Prompt: "A solitary figure standing on the railing of a tall pedestrian bridge at dusk, river below"
→ {"violates": true, "categories": ["self_harm"], "reason": "Person positioned on a bridge railing above water = suicide-method imagery (jumping)."}

Prompt: "A motionless figure floating face-down in a calm dark lake at night"
→ {"violates": true, "categories": ["self_harm"], "reason": "Motionless body face-down in water = drowning / suicide aftermath."}

Prompt: "A kitchen with all gas burners turned on without flames, a person seated unmoving at the table, dim light"
→ {"violates": true, "categories": ["self_harm"], "reason": "Gas burners on without flames + unmoving person = asphyxiation / suicide-method scene."}

Prompt: "A solitary figure standing at the edge of a foggy cliff, back turned to the camera, gloomy sky"
→ {"violates": true, "categories": ["self_harm"], "reason": "Person positioned at cliff edge with somber framing = suicide-ideation / jumping imagery."}

Prompt: "A blurred figure at a dim desk lit only by a lamp, looking down at a written list of options on a notebook"
→ {"violates": true, "categories": ["self_harm"], "reason": "Written list of options at a dim desk in this context = suicide-planning imagery."}

Prompt: "Muscular adult man in tight leather briefs flexing in front of a full-length mirror"
→ {"violates": true, "categories": ["sexual"], "reason": "Suggestive attire (tight briefs) + body-focused posing = sexual content."}

Prompt: "Razed rural village, smoldering thatched homes, civilians lying motionless in dirt road, soldiers walking away with torches"
→ {"violates": true, "categories": ["violence"], "reason": "Razed village with motionless civilians and torch-wielding soldiers = war atrocity, regardless of 'grim documentary tone'."}

Prompt: "Wall graffiti combining a stylized Sonnenrad black-sun symbol with the numbers 1488"
→ {"violates": true, "categories": ["hate"], "reason": "Sonnenrad and 1488 are recognized neo-Nazi hate codes."}

Prompt: "A cat sitting in a sunbeam by a window, photorealistic"
→ {"violates": false, "categories": [], "reason": "Benign scene, no disallowed content."}

Prompt: "A bowl of ramen with steam rising, food photography"
→ {"violates": false, "categories": [], "reason": "Benign food photography."}"""


CONTENT_FILTER_EDIT_SYSTEM = """You are a strict content-policy classifier for an IMAGE-EDITING system following Microsoft Responsible AI guidelines. You are shown one or more SOURCE IMAGES and a text EDIT INSTRUCTION. Decide whether this edit request must be blocked. Be CONSERVATIVE.

# CORE RULE
Block the request (violates=true) if EITHER the source image(s) ALREADY depict disallowed content, OR the edit instruction asks to PRODUCE disallowed content. You are NOT permitted to use any "but it's just an edit / the change is minor / the photo is generic / artistic / documentary / fictional" rationalization to lower a true violation to false. Judge BOTH the picture(s) in front of you AND the instruction.

# DECISIVE RULE (READ CAREFULLY)
If you can RECOGNIZE or NAME the subject of a source image as a real public figure (a specific real person), OR as a copyrighted/franchise character (e.g. "this is Pikachu", "this is a Pokémon", "this is Mario", "this is a fictional cartoon character from a franchise"), then that recognition IS ITSELF the violation. You must set violates=true. Do NOT then reason "but the requested edit is only a background/style/color change, so it's fine" — editing the image at all reproduces the protected likeness/IP, so ANY edit of it is disallowed. Naming it = blocking it.

# CATEGORIES (block if ANY applies to the source image OR the requested result)

- "sexual": the source image shows or the edit would create nudity, sexual acts, exposed/again suggestive intimate body parts, or scenes strongly implying nudity (topless, bare, naked, lingerie, wet see-through clothing, bathing/showering, bedroom intimacy). If the source photo is already NSFW/sexual, BLOCK regardless of how benign the instruction is.

- "public_figure": the source image DEPICTS a recognizable real named person — a politician, head of state, royalty, celebrity, musician, actor, athlete, or other public figure (e.g. Trump, Biden, Xi Jinping, Putin, Modi, Obama, Zelenskyy, Netanyahu, Musk, Taylor Swift, Beyoncé, Cristiano Ronaldo, Messi, etc.). Editing a photo of a real recognizable public figure is DISALLOWED regardless of the instruction (changing background, clothing, style all count). If you recognize the person in the photo as a real public figure, BLOCK.

- "copyright": the source image shows OR the edit would insert a copyrighted character, branded property, named franchise icon, or company logo/mascot (Disney, Star Wars, Marvel, DC, Nintendo/Mario/Zelda, Pokémon/Pikachu, Minions, Simpsons, Harry Potter, Spider-Man, Batman, Godzilla, Hello Kitty, etc.). Editing an image OF a copyrighted character is DISALLOWED regardless of how "stylistic" or "generic" the edit is.

- "hate": hate symbols or dehumanizing/mocking imagery of protected groups present in the source or requested (swastika, KKK, Sonnenrad/black-sun, 1488, SS bolts, antisemitic/anti-Muslim/homophobic/transphobic/ableist/misogynistic/racist caricature).

- "self_harm": the source depicts or the edit implies suicide, self-injury (cutting/scarring/burning), asphyxiation/hanging/jumping/drowning setups, suicide notes/aftermath, or eating-disorder / pro-ana imagery.

- "violence": the source depicts or the edit would add gore, weapon attacks, torture, executions, war atrocities, mass-casualty scenes, brutal assault, or animal cruelty. "Documentary / grim / cinematic / historical" framing does NOT exempt it.

# OUTPUT
Respond with STRICT JSON ONLY (no markdown, no preamble, no commentary):
{"violates": true_or_false, "categories": ["sexual" | "hate" | "self_harm" | "violence" | "copyright" | "public_figure"], "reason": "<one short sentence>"}

# RULES
- Set "violates": true if ANY category applies to the source image(s) OR the requested edit.
- If the source photo shows a real recognizable public figure, a copyrighted character, or NSFW content, BLOCK even when the instruction itself is innocuous (e.g. "change the background", "make it anime style", "remove the hat").
- Do NOT rationalize ("but it's only an edit / fictional / artistic / documentary / educational").
- Do NOT explain in any text outside the JSON object.

# EXAMPLES (the source image is described in [brackets]; your decision must match)

[source: a yellow Pokémon character (Pikachu) standing in a grassy field] + instruction "change the background to a beach"
→ {"violates": true, "categories": ["copyright"], "reason": "Source image is Pikachu, a copyrighted Pokémon character; editing it reproduces protected IP regardless of the innocuous background change."}

[source: Mario, red hat and blue overalls] + instruction "make it anime style"
→ {"violates": true, "categories": ["copyright"], "reason": "Source depicts Mario, a copyrighted Nintendo character; any edit of it is disallowed."}

[source: a photo of Donald Trump at a podium] + instruction "change the background to a beach"
→ {"violates": true, "categories": ["public_figure"], "reason": "Source depicts a recognizable real public figure; editing his photo is disallowed regardless of the edit."}

[source: a generic golden retriever puppy on grass] + instruction "change the background to a beach"
→ {"violates": false, "categories": [], "reason": "Ordinary non-sensitive photo and an innocuous edit."}

[source: an ordinary unknown man in a suit on a street] + instruction "change the background to a beach"
→ {"violates": false, "categories": [], "reason": "Not a recognizable public figure; ordinary person, innocuous edit — allowed."}"""


CATEGORY_DISPLAY = {
    "sexual": "Sexual content",
    "hate": "Hate / unfair imagery",
    "self_harm": "Self-harm",
    "violence": "Violence / gore",
    "copyright": "Copyright / IP character",
    "public_figure": "Real-person likeness",
}


@dataclass
class FilterVerdict:
    violates: bool
    categories: list[str]
    reason: str
    raw: str = ""

    def banner(self) -> str:
        if not self.violates:
            return ""
        cat = ", ".join(CATEGORY_DISPLAY.get(c, c) for c in self.categories) or "policy violation"
        return f"🚫 **Content Filter:** Blocked — `{cat}` · {self.reason}"


def _extract_json_object(text: str) -> dict:
    """Pull the first balanced top-level JSON object out of a possibly-wrapped string."""
    if not text:
        raise ValueError("empty response")
    # Strip code fences
    if text.lstrip().startswith("```"):
        text = text.strip().strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in: {text[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON object: {text[start : start + 120]!r}")


def check_prompt(model, prompt: str, max_new_tokens: int = 160) -> FilterVerdict:
    """Back-compat wrapper around the mandatory text-encoder screener.

    The policy check now lives on the text encoder
    (:meth:`TextEncoder.screen_text`) so it runs on the same Qwen3-VL weights
    that produce the diffusion conditioning and is FAIL-CLOSED. Kept here for
    external callers / tests that import ``check_prompt`` directly.
    """
    return model.txt_enc.screen_text(prompt, max_new_tokens=max_new_tokens)


@torch.no_grad()
def screen_mage_flow_prompt(
    text_encoder,
    processor,
    prompt: str,
    *,
    max_new_tokens: int = 160,
) -> FilterVerdict:
    """Run the official fail-closed policy prompt on the shared Qwen3-VL."""
    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "system",
            "content": CONTENT_FILTER_SYSTEM,
        },
        {
            "role": "user",
            "content": f"Prompt to classify:\n{prompt}",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(text_encoder.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    try:
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        output_ids = text_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        input_length = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(
            output_ids[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        parsed = _extract_json_object(raw)
        violates = parsed.get("violates")
        categories = parsed.get("categories")
        reason = parsed.get("reason")
        if (
            not isinstance(violates, bool)
            or not isinstance(categories, list)
            or not all(isinstance(item, str) for item in categories)
            or not isinstance(reason, str)
        ):
            raise ValueError("content checker returned an invalid schema")
        return FilterVerdict(violates, categories, reason, raw=raw)
    except Exception as error:
        # The official behavior is fail-closed: checker failure must not turn
        # into an unreviewed generation.
        return FilterVerdict(
            violates=True,
            categories=["safety_check_error"],
            reason=f"content safety check failed: {error}",
        )


@torch.no_grad()
def screen_mage_flow_edit_prompt(
    text_encoder,
    processor,
    prompt: str,
    reference_images: list[Image.Image],
    *,
    max_new_tokens: int = 192,
) -> FilterVerdict:
    """Run the official fail-closed multimodal policy check for an edit."""
    if not reference_images:
        return screen_mage_flow_prompt(
            text_encoder,
            processor,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    instruction = prompt.strip() or "(no textual instruction)"
    try:
        tokenizer = processor.tokenizer
        user_content = [{"type": "image"} for _ in reference_images]
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"There {'is' if len(reference_images) == 1 else 'are'} "
                    f"{len(reference_images)} source image(s) above. "
                    f"Edit instruction: {instruction}\nClassify this edit request."
                ),
            }
        )
        messages = [
            {"role": "system", "content": CONTENT_FILTER_EDIT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[rendered],
            images=reference_images,
            padding=True,
            return_tensors="pt",
        )
        device = next(text_encoder.parameters()).device
        generation_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "mm_token_type_ids",
            }
            and value is not None
        }
        input_length = generation_inputs["input_ids"].shape[1]
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        with _full_output_mode(text_encoder):
            output_ids = text_encoder.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        raw = tokenizer.decode(
            output_ids[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        parsed = _extract_json_object(raw)
        violates = parsed.get("violates")
        categories = parsed.get("categories")
        reason = parsed.get("reason")
        if (
            not isinstance(violates, bool)
            or not isinstance(categories, list)
            or not all(isinstance(item, str) for item in categories)
            or not isinstance(reason, str)
        ):
            raise ValueError("content checker returned an invalid schema")
        return FilterVerdict(violates, categories, reason, raw=raw)
    except Exception as error:
        return FilterVerdict(
            violates=True,
            categories=["safety_check_error"],
            reason=f"edit content safety check failed: {error}",
        )


@contextmanager
def _full_output_mode(hf):
    """Temporarily switch the Qwen3-VL encoder into FULL output mode so that
    ``.generate()`` sees ``.logits`` (the diffusion path uses embedding mode).
    Restores the original mode/skip flags on exit — side-effect free."""
    prev_mode = getattr(hf, "_output_mode", None)
    prev_skip = getattr(hf, "_skip_lm_head", None)
    try:
        if hasattr(hf, "set_output_mode"):
            try:
                hf.set_output_mode("full")
            except Exception:  # noqa: BLE001
                if prev_skip is not None:
                    hf._skip_lm_head = False
                if prev_mode is not None:
                    hf._output_mode = "full"
        elif prev_skip is not None:
            hf._skip_lm_head = False
        yield
    finally:
        try:
            if prev_mode is not None and hasattr(hf, "set_output_mode"):
                hf.set_output_mode(prev_mode)
            if prev_skip is not None:
                hf._skip_lm_head = prev_skip
        except Exception:  # noqa: BLE001
            pass


def check_edit(model, prompt: str, ref_images, max_new_tokens: int = 192) -> FilterVerdict:
    """Back-compat wrapper around :meth:`TextEncoder.screen_edit`.

    Classifies an image-EDIT request considering BOTH the source image(s) and
    the instruction (multimodal Qwen3-VL), FAIL-CLOSED. Kept for external
    callers / tests that import ``check_edit`` directly.
    """
    return model.txt_enc.screen_edit(prompt, ref_images, max_new_tokens=max_new_tokens)


def make_refusal_image(
    verdict: FilterVerdict,
    height: int = 1024,
    width: int = 1024,
) -> Image.Image:
    """Return a placeholder image to display when the prompt is blocked.

    A plain white blank image — no text, no category/reason surfaced.
    """
    return Image.new("RGB", (width, height), color=(255, 255, 255))
