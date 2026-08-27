# Open Parking AI — Vehicle ID

Identify a vehicle from camera frames, and say how sure you are.

This is a **replacement for an LPR unit**. It runs on its own, on the same
device or the same LAN as whatever consumes it, and it identifies with the
internet down. It is INTEGRATED, not embedded: there is one contract, and Open
Parking AI's own lane controller is an ordinary client of it — the same door a
third party uses. No in-process path is reserved for us.

```
captures ─▶ engine ─▶ READ record ─┬─▶ answer    plate + confidence
                                   └─▶ fallback  not sure, and saying so
```

## Never wrong silently

The property everything else here exists to protect. When the engine is not
confident enough to stand behind a read, it says so: `outcome: "fallback"` is a
**first-class answer**, not an error, not an exception, not an empty response.

A system that is wrong loudly is recoverable. A system that is wrong silently
bills a stranger's car to somebody else, and nobody finds out.

Three rules follow from it, and each is enforced by a test rather than a promise:

- **A field that was not MEASURED is null.** `make`, `model` and `color` are null
  today because no slice measures them yet — not because the car had none. A
  plausible guess in one of those fields would be indistinguishable from a
  measurement to everything downstream.
- **The operating threshold travels with the read.** `threshold_applied` is in
  every record, so a consumer can see the operating point that produced the
  outcome. It is **measured**, not chosen — see the harness below.
- **An unknown `schema_version` is refused, not half-read.** A build that does
  not recognise the record it is handed says so rather than guessing which
  fields still mean what they used to.

## Try it with no system at all

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,engine]'
python -m vehicle_id.plates.train        # weights are NOT committed; build your own
vehicle-id read ./some-photos
```

```
  ANSWER    7ABC123      confidence 0.996  car-01.jpg
  FALLBACK  —            confidence 0.412  car-02.jpg
            below the measured operating point (0.990) — not an error, an answer
```

## Integrating it

```sh
vehicle-id serve --push-to http://your-system.local/reads
```

| | |
|---|---|
| `POST /v1/reads` | submit captures, get the record back — **synchronous**, for a caller with a vehicle in front of it |
| `GET /v1/reads/last` | the most recent record |
| `GET /v1/reads?since=N` | everything after a cursor, for a consumer catching up |
| `GET /v1/health` | engine, version, weights digest, operating point |
| push | every record POSTed to your URL as it happens, with a durable local queue |

Push never loses a record: it is written to disk **before** delivery is
attempted and removed only once you have acknowledged it. Re-sends are therefore
normal — `read_id` is stable across attempts, and you are expected to
deduplicate on it. A `4xx` from you is treated as a refusal and dropped rather
than retried forever, because retrying poison blocks everything behind it.

The full record, its field-by-field meaning and the compatibility rules are in
[docs/CONTRACT.md](docs/CONTRACT.md).

**Local by design.** The service binds to loopback unless you tell it
otherwise. There is no cloud in the identification path and there will not be
one: a lane has to identify when the internet is down.

**Deferred, and named so their absence is not read as an oversight:**
RS-232/RS-485 serial, Wiegand, dry-contact relay, MQTT and ONVIF events. They
cannot be tested honestly without hardware, so they are not claimed.

## What is actually measured

The recogniser is **ours** — a small CRNN+CTC trained on generated plates. That
was a licensing decision before it was a technical one: no public US
plate-reading dataset permits commercial use (see
[docs/EVAL_DATA.md](docs/EVAL_DATA.md)), and every third-party model checked
ships permissive *code* with **no stated licence for its weights**. Generating
the training data removes both problems and makes the weights our own asset.

```sh
python scripts/eval_plates.py     # every number quoted anywhere comes from here
```

Measured on **synthetic** plates, 200 per rung across a 10-rung degradation
ladder:

| | rungs 0–7 exact | rung 8 | rung 9 | ms/plate |
|---|---|---|---|---|
| ours (CRNN+CTC) | 98.0–100% | 88.5% | 68.5% | 2.0–7.1 (CPU, load-dependent) |
| baseline (RapidOCR PP-OCRv3) | 10.5–98.5% | — | 50.0% | 208.7 |

> ### Real-plate accuracy is NOT MEASURED
>
> Every number above measures **the generator**, not the world. The plates were
> made with OpenCV's built-in fonts; real plates use embossing typefaces we
> neither have nor could redistribute, and that is the largest domain gap there
> is.
>
> **No real-world accuracy claim is made, and none will be, until a physical
> bench exists** — our own cameras, our own vehicles, kept local. The harness
> prints real-plate accuracy as `NOT MEASURABLE` with the reason, rather than
> quoting the synthetic number and hoping.
>
> This paragraph travels with the engine wherever it ships. If you are
> evaluating this against a commercial LPR unit, that unit's published number
> and the number above are not comparable, and treating them as comparable is
> the mistake this box exists to prevent.

**The recogniser is accurate AND overconfident**, which matters more than the
accuracy: mean confidence barely moves across the ladder while accuracy falls.
So the operating point is measured, not chosen — **0.99**, the cheapest
threshold whose silent-wrong rate falls below 1% (0.87% wrong-but-answered,
30.9% sent to fallback). At a naive 0.85 the same model answers wrongly 4.45% of
the time. That is the whole argument for shipping the threshold inside the
record.

## What it does not do yet

Make, model, colour, distinguishing appearance and re-identification are the
next slice, written against this contract. Their fields exist in the record and
are null. Nothing here invents them.

## Runs with no engine at all

The contract, the service, the push queue and the CLI are standard library only.
A consumer integrating against this product installs it and imports
`vehicle_id.contract` with no torch and no OpenCV — and CI proves that claim by
running the whole test suite in an environment where neither is installed.

```sh
pip install -e '.[dev]'    # no engine
pytest
```

## Licence and contributing

AGPL-3.0-or-later — see [LICENSE](LICENSE). Contributions require a signed CLA;
see [CONTRIBUTING.md](CONTRIBUTING.md).

Weights and models used here are permissively licensed only, and weights are
downloaded or built, never committed.

---

Built by 72 Knots Method by 72Knots.ai
