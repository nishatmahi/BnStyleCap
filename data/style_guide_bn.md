# Style guide: Bangla humorous and romantic captions

One humorous and one romantic caption per image, written from that image's factual captions.
Output is keyed by image id: `image_id <TAB> humorous <TAB> romantic`.

## Rules for both styles

1. **Grounded.** Keep the main subject and main action of the factual captions. Never add a person, animal or object that no factual caption mentions.
2. **Length.** 8 to 18 words, one sentence, ends with "।".
3. **Language.** Standard spoken Bangla (চলিত). No English words unless the factual caption has them (e.g. brand names). No image ids, numbering, emojis or quotation marks.
4. **No templates.** Do not reuse a fixed ending or a fixed comparison. Vary sentence shape.
5. **Not a copy.** The styled caption must not be the factual caption with one phrase inserted.

## Humorous

Mix these kinds of humor, roughly evenly:

| Kind | Example idea |
|---|---|
| Exaggeration | the dog acts like it won an award |
| Objects/animals with human thoughts | the umbrella is tired of the rain |
| Ironic comment on the scene | "this is what people call a relaxing holiday" |
| Everyday Bangladeshi life | bus rush, exams, relatives, load shedding, traffic, mothers' orders |
| Wordplay / light self-deprecation | playful twist on a word in the scene |

Limits:
* The "অবিকল/একদম/ঠিক ... এর মতো/কায়দায়/ভঙ্গিতে" simile pattern in at most 1 in 5 captions.
* Any single comparison figure (e.g. কুম্ভকর্ণ) at most 15 times in the whole dataset.
* Gentle humor only. No mocking religion, body, gender, ethnicity or disability.

## Romantic

* Feelings: warmth, tenderness, longing, nostalgia, companionship, quiet joy.
* The feeling must come from something visible in the scene (hands held, rain on glass, an old man with a child, an empty bench at sunset).
* Vary endings. Do not end most captions with a feeling noun (মায়া, স্নেহ, প্রীতি, মুগ্ধতা, আবেশ ...). Each such noun at most 3% of captions.
* Sometimes address "তুমি", sometimes describe in third person.
* Tender, not sexual.

## Added after pilot review (2026-09-27)

* **No new templates.** "যেন" in at most 25% of captions per style. The pattern "এমন ... যেন" in at most 10%.
* **Romantic endings.** Phrases with "মন" (মন ভরে যায়, মনে পড়ে, মন ছুঁয়ে যায়, মনটা ...) in at most 15%.
* **Humor mix.** At least 30% of humorous captions use everyday Bangladeshi life (bus, rickshaw, traffic, exams,
  mothers, relatives, load shedding, bazar, tea stall). Also use irony, objects with thoughts, wordplay.
  Exaggeration alone in at most 40%.
* **Romantic, not only nostalgic.** Use warmth, love, longing, companionship between people or toward a place.
  At least 20% address "তুমি/তোমার". Nostalgia ("শৈশব মনে পড়ে") in at most 15%.
* **No invented setting.** Do not add time of day (সকাল, বিকেল, সন্ধ্যা, রাত, ভোর), weather or season unless a
  factual caption mentions it. Do not change the action (waiting must not become running).

## Added after pilot v2 (2026-09-27)

* Humorous: "বুঝি" in at most 15%, "ভাবখানা / ভাব দেখে" in at most 10%, "এমন" in at most 20% of captions.
* Keep the action of the factual captions (a waiting crowd is not "pushing").

## Automatic checks after every batch

* simile-pattern rate ≤ 20%, top comparison figure ≤ 15 uses
* length 8 to 18 words, no Latin letters unless in the factual captions, no digits-only ids
* no exact duplicates within a style
* যেন ≤ 25%, এমন...যেন ≤ 10%, romantic মন-phrases ≤ 15%, তুমি/তোমার ≥ 20% of romantic
* no time-of-day word unless in the factual captions
* at least one content noun shared with the factual captions (grounding)
* humorous and romantic caption for the same image are not near-identical

Captions failing any check are rewritten.

## Added after Gemini flash-lite test (2026-09-27)

* Humorous: "মনে হয়" (e.g. "দেখে মনে হয়") in at most 15%. The pattern "এমন ... যেন" in at most 10% (checked per part).
* Romantic: "অদ্ভুত" (e.g. "এক অদ্ভুত ভালোলাগা") in at most 10%, "মূর্ছনা" in at most 5%.
