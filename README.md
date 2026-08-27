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
- **The operating threshold travels with the read**, and it was measured for
  *those exact weights*. `threshold_applied` is in every record. The engine
  **refuses to start** on weights whose operating point nobody has measured,
  rather than stamping a constant onto records a different model produced.
- **"Is a vehicle there" is asked first, and answered separately.** Presence is
  its own field with its own confidence and is never folded into the identity
  confidence. It has **three** states: `true`, `false`, and `null` for NOT
  MEASURED — a lane with no reference view of its empty tarmac behaves exactly
  as it did before this stage existed rather than refusing everybody. A record
  with `presence: false` **cannot** carry an identity; the contract refuses to
  build one.
- **A batch that disagrees with itself is a fallback.** Several captures are of
  one vehicle; if more than one reads a plate confidently and they disagree,
  the engine does not pick the higher score. That is a tailgater or a second
  car in frame, and the honest answer is that it does not know which one is at
  the barrier.
- **An unknown `schema_version` is refused, not half-read.** A build that does
  not recognise the record it is handed says so rather than guessing which
  fields still mean what they used to.

## Try it with no system at all

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,engine]'
python -m vehicle_id.plates.train        # weights are NOT committed; build your own
python scripts/eval_plates.py --write-operating-point   # measure before you trust
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
| `GET /v1/health` | engine, version, weights digest, operating point, push state — `degraded` when a read was lost |
| push | every record POSTed to your URL as it happens, with a durable local queue |

Push never loses a record: it is written to disk **before** delivery is
attempted and removed only once you have acknowledged it. Re-sends are therefore
normal — `read_id` is stable across attempts, and you are expected to
deduplicate on it. A `4xx` from you is treated as a refusal and dropped rather
than retried forever, because retrying poison blocks everything behind it.
Anything outstanding when the process stopped is delivered at startup, before
the first vehicle of the day, and retried on a timer rather than waiting for
the next car.

Send `request_id`, or an `Idempotency-Key` header, and a re-sent submission
returns the same record instead of becoming a second vehicle.

**The service is not authenticated and the queue file is trusted local state.**
Keep both where you would keep a credential — see
[docs/CONTRACT.md](docs/CONTRACT.md).

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

> ### It has no rejection capability, and that is measured too
>
> This recogniser reads text out of anything. Measured on 300 images per case,
> with the reference weights:
>
> | shown | read something | mean confidence | cleared 0.99 |
> |---|---|---|---|
> | uniform sensor noise | 300/300 | 0.83 | **2.3%** |
> | blurred noise | 300/300 | 0.61 | 0% |
> | flat black / white / grey | 300/300 | 0.52–0.63 | 0% |
>
> So **confidence alone cannot tell a plate from snow.** A dead or noisy camera
> feed produces a confident plate on roughly one single frame in forty.
>
> Reading several captures of one vehicle mitigates it, because noise reads
> *differently* every frame and the batch then disagrees with itself: **2.3% for
> one capture, 0.3% for three.** The **presence gate** closes the rest — with a
> reference view of the empty lane in front of it, the same measurement is
> **0.0% at one capture and 0.0% at three.**
>
> A consumer with its own second gate is not exposed to the residual: measured
> against a lane holding a 200-plate permit list, 2 confident noise reads out of
> 300 opened **0** barriers, because a random string matches no rule. A garage
> configured to allow everything it can identify has no such gate, and for that
> configuration this number is the risk.

**The recogniser is accurate AND overconfident**, which matters more than the
accuracy: mean confidence barely moves across the ladder while accuracy falls.
So the operating point is measured, not chosen — **0.99**, the cheapest
threshold whose silent-wrong rate falls below 1% (0.87% wrong-but-answered,
30.9% sent to fallback). At a naive 0.85 the same model answers wrongly 4.45% of
the time. That is the whole argument for shipping the threshold inside the
record.

## The presence gate

```sh
vehicle-id serve --empty-lane ./lane-empty.png
```

The question is **"is a VEHICLE present"**, not "is a plate present". Those are
different questions and conflating them breaks the product in both directions: a
car with a filthy, damaged or missing front plate is a legitimate entry and must
be admitted, and a metal object held over an inductive loop is not a car and must
receive nothing at all — no ticket, no session, no barrier.

**No model, no weights, no dataset**, and that is a licensing decision before a
technical one. Every general-purpose detector worth using is COCO-trained, and
COCO's images are not the consortium's to license: the annotations are CC BY 4.0
but the images are Flickr's, individually, with users accepting full
responsibility. torchvision says the same of its own weights — they "may have
their own licenses … derived from the dataset used for training", and working out
whether you may use them is your problem. That is the trap that disqualified
Ultralytics and OpenALPR before a line was written. And the escape the recogniser
used — generate the training data — is not available: a plate is a rendered
rectangle with a parameterisable font, which is why that generator works. A car
is not.

So it is answered the way a fixed camera on fixed tarmac makes possible: how much
of the scene changed, in one contiguous region. A vehicle fills a large part of
the frame. A person holding a plate does not, and neither does rain, a bird or a
shadow — scattered change across 45% of the frame reads as **absent**, because
only the largest connected region counts.

> **What is measured and what is not.** This gate can be shown to reject sensor
> noise, to reject a plate-sized object, to reject scattered speckle, and to
> admit a plate the recogniser reads perfectly — all of which are in the test
> suite. Its behaviour on **real vehicles at a real lane is NOT MEASURED**, and
> will stay that way until the physical bench exists. The occupancy threshold is
> an **assumption**, not a measurement: it cannot be measured without lane
> footage, so it is a documented, configurable number rather than a constant
> presented as a finding.

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
