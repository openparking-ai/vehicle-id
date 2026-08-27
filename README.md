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
> *differently* every frame and the batch then disagrees with itself:
> <!--m:noise.one_capture-->1.9% mean, 0.0-4.0% across 12 seeds x 150 reads<!--/m--> at one capture,
> <!--m:noise.three_capture-->1.4% mean, 0.7-3.3% across 12 seeds x 150 reads<!--/m--> at three.
> **It does not reach zero.** Measured on weights
> <!--m:noise.weights-->sha256:0de21983b58b0ecd<!--/m--> — a rate is meaningless
> without the model it describes, so it is published with one.
>
> With a reference view of the empty lane in front of it, the presence gate
> takes the same measurement to <!--m:noise.gated_one_capture-->0.0% (12 seeds x 150 reads, 1800 in total, none)<!--/m-->
> at one capture and <!--m:noise.gated_three_capture-->0.0% (12 seeds x 150 reads, 1800 in total, none)<!--/m--> at three
> — a dead feed is not a picture, and the gate stops the read before the
> recogniser sees it. That is measured **against sensor noise**, which is one of
> the two ways a camera fails; see the deployment note below for what it does
> and does not cover.
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
shadow — because only the largest connected region counts.

**Structure, not brightness.** A window counts as changed when it has lost more
than <!--m:gate.min_structural_change-->0.5<!--/m--> of its structural agreement
with the reference — the contrast and structure terms of SSIM, with the
luminance term deliberately dropped. Deciding on the size of an intensity
difference instead is blind to a vehicle the same brightness as the ground:
under the previous measure a car occupying 43.75% of the frame — nearly three
times the occupancy floor — read `presence: false` at 0.97 confidence for any
body shade within about 30 grey levels of the tarmac. That is an ordinary grey
car on grey asphalt, `false` is the value that ends a transaction, and moving
the threshold only moves the band. A car at the tarmac's own luminance still
occludes the tarmac's texture, and that is what is measured now.

### Three states, and the two of them that are not a verdict

`true`, `false`, `null`. `false` is a **measurement** — the lane is visible, it
matches the reference, and there is nothing on it — and it is the only value
that ends a transaction before it starts. So it is the only one the gate will
say from a measurement it actually has. Everything it cannot see is `null`, and
`null` puts the lane back to the behaviour it had before this stage existed: a
ticket and a human.

The first version of this gate did not hold that line. It measured raw intensity
against the reference and called any large contiguous change a vehicle, so a
dead camera, a blown exposure, dusk against a daylight reference and sun on
tarmac all reported `presence: true`, three of them at confidence 1.0, with
nothing in frame. None of them opened a barrier — the recogniser's measured
operating point held — but the gate was asserting a positive measurement of
something it had not measured, which is the failure this project exists to not
have. Three things changed:

- **A frame carrying no picture can never produce a verdict.** Flat black, flat
  white, a taped lens, a dead feed. Caught by variance and by whether
  neighbouring pixels agree at all, reported as a named camera fault, and the
  read is stopped — the recogniser has no rejection stage and would mint a
  confident plate out of it.
- **A change in the light is cancelled before a change in the scene is looked
  for.** One gain and one offset are fitted between the capture and the
  reference and undone. An empty lane now reads `false` from
  <!--m:exposure.range-->light level 20 to 250 against a reference captured at 90<!--/m-->, at
  <!--m:exposure.holds-->every level tested<!--/m--> tested.
- **A change that fills the frame is `null`, not `false`.** Above
  <!--m:gate.max_occupancy-->90%<!--/m--> occupancy the scene changed rather
  than something arriving in it — a camera that was knocked, or a van close
  enough to fill the view. An upper bound that answered `false` would refuse the
  vans, trucks and buses that legitimately fill an entry lane, and at the lane
  that is no ticket, no session and no vend for a customer who is really there.

> ### UNVALIDATED AGAINST REAL VEHICLES. OPT-IN, AND OFF UNTIL YOU TURN IT ON.
>
> **Every number in this section describes synthetic scenes.** Rectangles on a
> drawn lane, photographed by nothing. No frame of real footage has ever been put
> through this gate. It is **off unless you configure a reference view** — without
> one presence is `null` and the lane behaves exactly as it did before this stage
> existed — and a garage that switches it on is choosing an unvalidated
> measurement, knowingly. That is the whole basis on which it ships.
>
> Three rounds of adversarial review each found this gate measuring something
> other than presence, or claiming something it had not measured. Each was found
> by inventing a probe nobody had thought of. That process does not terminate on
> synthetic data. Real lane footage is the only thing that ends it, and it does
> not exist yet.

Everything from here to the end of this section is **generated from
`docs/measured/presence.json`** by `scripts/eval_presence.py --update-docs`, and
`tests/test_measured_docs.py` fails if a word of it drifts from the measurement.
The sentences are generated and not merely the numbers, because a correct number
beside a false sentence is exactly how the last round's claim survived: the span
saying `false` held to 5% streak coverage was right, and the hand-written words
"and stops answering above that" next to it were not.

<!--mb:presence.separation-->
**The matrix, unedited.** 108 cells sweeping vehicle/ground contrast through the exactly-invisible case, ground texture, the vehicle's own surface grain, and a headlight pool on the floor. Each cell carries both scenes — the vehicle and the empty lane beside it — because a measure that answered one way for everything would pass a one-sided sweep perfectly.

| configuration | cells | vehicle seen | vehicle refused | vehicle not measured | empty called empty | empty read occupied | margin | separates |
|---|---|---|---|---|---|---|---|---|
| ground texture 0.25, headlights off | 18 | 18 | 0 | 0 | 0 | 18 | 0.045 | **no** |
| ground texture 0.25, headlight pool x3 | 18 | 16 | 0 | 2 | 4 | 14 | 0.247 | **no** |
| ground texture 1, headlights off | 18 | 18 | 0 | 0 | 18 | 0 | 0.425 | yes |
| ground texture 1, headlight pool x3 | 18 | 18 | 0 | 0 | 18 | 0 | 0.446 | yes |
| ground texture 2, headlights off | 18 | 18 | 0 | 0 | 18 | 0 | 0.424 | yes |
| ground texture 2, headlight pool x3 | 18 | 18 | 0 | 0 | 18 | 0 | 0.440 | yes |

It separates vehicle from empty in 4 of 6 configurations, with a worst-case occupancy margin of 0.42.
It does **not** separate in ground texture 0.25, headlights off: 18 of 18 empty lanes read as occupied.
It does **not** separate in ground texture 0.25, headlight pool x3: 14 of 18 empty lanes read as occupied.

**A vehicle is admitted in 106 of the 108 cells**, including at the ground's exact luminance, where the vehicle and the ground it stands on are the same brightness. None of them was refused. In the remaining 2 presence is `null` and the gate reports a camera fault: `reference_not_recognised` in 2. A cell that is not measured is not a refusal — the lane falls back to a ticket and a human — but it is a frame with a car in it that this gate answered nothing about, and the reason it gave names equipment.
<!--/mb-->

<!--mb:presence.texture-->
**Ground with no texture of its own is NOT MEASURED, never `false`.** The comparison asks whether a window still looks like the same piece of ground, so ground carrying nothing to recognise leaves it nothing to work with. Below 1.5 grey levels of typical local texture the gate declines to answer.

The matrix's own ground never gets near that floor — its texture axis bottoms out at 3.821 grey levels (texture 0.25 → 3.821, texture 1 → 9.194, texture 2 → 17.46), because the sensor's own grain is most of it.

Sealed or painted concrete under a clean sensor is a different scene: it measures 0.67 grey levels, the gate returns `null` with no camera fault raised, and it says why — "the reference view's local texture is 0.67 grey levels, below 1.5; this ground carries nothing for a structural comparison to recognise".

**This matters more than the number suggests.** 0.67 grey levels against a 1.5 floor is a surface this gate declines to answer on at all. **Whether that describes a given entry is a property of that entry** — the operator can photograph its floor and score it by the mapping in `docs/EVAL_DATA.md`; how many entries look like it is not something this project has measured. NOT MEASURED on any real frame (0 have ever been through this gate), so how much texture a real covered entry carries is an open question, and the remedy if it carries too little is physical — paint markings, add a textured strip in view.
<!--/mb-->

<!--mb:presence.weather-->
**Weather, measured on three scenes at every coverage** — an empty lane, a vehicle, and the metal plate the gate exists to refuse. The number beside each verdict is the measured occupancy, over 8 coverages.

| streak coverage | empty lane | vehicle | metal plate |
|---|---|---|---|
| 5% | `false` 0.021 | `true` 0.519 | `false` 0.047 |
| 10% | `true` 0.428 | `true` 0.666 | `true` 0.431 |
| 15% | `true` 0.612 | `true` 0.724 | `true` 0.615 |
| 20% | `true` 0.825 | `true` 0.878 | `true` 0.825 |
| 25% | `true` 0.890 | `null` — | `true` 0.890 |
| 30% | `null` — | `null` — | `null` — |
| 35% | `null` — | `null` — | `null` — |
| 45% | `null` — | `null` — | `null` — |

**Three bands, not two.** `false` up to 5% of the frame in streaks; from 10% an **empty lane reads as OCCUPIED**, at up to 0.99 confidence; from 30% the gate declines to answer at all.

The band from 10% is the one to read. `presence: true` with `outcome: "fallback"` tells a lane controller that a car is there and could not be identified, and this contract says refusing it is a bug in your integration — so in that band a conforming lane issues a ticket and raises an attendant for a car that is not there.

**And the fraud is admitted with it.** The metal plate on the loop — the case this gate exists for — is correctly refused up to 5% coverage and then **transacts from 10%**, on the same streaks. In that band the gate does not merely lose the ability to say `false`; it issues the ticket for the exact scene it was built to refuse.

This is a measured REGRESSION against the intensity measure that preceded it, which called heavy rain an empty lane correctly. It is recorded rather than argued away. **Whether it reaches a given entry depends on whether bright streaks break up the camera's view of the ground, which rain and snow both do; what was measured is how much of the frame they cover, not what produced them.** The operator can see that and this project cannot count it. NOT MEASURED on any real frame (0 have ever been through this gate), and no frequency is claimed either way. Across the sweep, 0 of 8 vehicle scenes were refused.
<!--/mb-->

<!--mb:presence.headlight-->
**Headlights on the floor.** A car with its beams on throws them into frame before the car itself arrives — a large change in the scene caused by a vehicle that is not yet the vehicle. Measured over 8 pools, with and without the car that cast them. `pool` is the beam's peak as a multiple of ambient ADDED to it, so the table states peak = 1 + pool.

| beam pool, peak x ambient | empty lane (car not yet in frame) | vehicle |
|---|---|---|
| x1 | `false` 0.001 | `true` 0.428 |
| x1.5 | `false` 0.000 | `true` 0.442 |
| x2 | `false` 0.000 | `true` 0.447 |
| x3 | `false` 0.001 | `true` 0.453 |
| x4 | `true` 0.205 | `true` 0.498 |
| x5 | `true` 0.335 | `true` 0.540 |
| x7 | `true` 0.477 | `true` 0.639 |
| x9 | `true` 0.804 | `true` 0.667 |

An empty lane holds at `false` up to a pool of x3 ambient and reads as OCCUPIED from x4 — the beams of a car that has not arrived open a transaction for it.

0 of 8 vehicle scenes were refused. **The model is a limitation of these numbers**: multiplicative pool on a matte floor; no specular glare, no beam cut-off. A gloss or wet floor at night is a specular scene and this is a matte one. NOT MEASURED on any real frame (0 have ever been through this gate).
<!--/mb-->

<!--mb:presence.safety-->
**The one thing that holds everywhere measured.** 0 wrongful refusals in 124 scenes containing a vehicle: 108 matrix cells, 8 weather coverages and 8 headlight pools, each measured with a vehicle in the frame. `false` is the only value that ends a transaction, and no scene measured produced it for a frame with a vehicle in it. Where this gate fails it fails to `null` — a ticket and a human.

Every one of those 124 scenes is a drawn rectangle on a drawn lane — NOT MEASURED on any real frame (0 have ever been through this gate). The claim is that the measure holds across everything that has been put through it, not that everything has been put through it.
<!--/mb-->

<!--mb:presence.conflation-->
**One reason covers 4 unrelated conditions, and this release cannot tell them apart.** `reference_not_recognised` is reported for all of the following:

- a capture that is not a view of this lane: a camera knocked out of alignment, or a scene rebuilt overnight
- a vehicle close enough to fill the frame
- an ordinary vehicle arriving on low-texture ground under a beam pool
- heavy weather

It is published under `camera_faults` in `GET /v1/health`. That is right for 1 of the 4 — a capture that is not a view of this lane — and wrong for the other 3, where nothing is broken. **Do not read this reason as a confirmed equipment fault** — read it as "the capture no longer matches the reference, for one of several reasons this build cannot separate". Separating them needs a measurement this release does not make, and inventing one would be guessing; naming the conflation is the honest thing available now.

**One of those conditions is a car arriving.** 2 of the 108 separation-matrix cells put an ordinary vehicle — 44% of the frame, not one filling it — in front of the camera and got `reference_not_recognised` back. Each of those cells, in full: ground texture 0.25, headlight pool x3 — contrast 2.05 / surface grain 0, contrast 2.05 / surface grain 0.02. The gate counts that under `camera_faults`, so an arriving car pages a technician about a working camera. NOT MEASURED on any real frame (0 have ever been through this gate): how often a real entry lands in one of these configurations is not known, and these are drawn rectangles. What is known is that the reason cannot be read as equipment on its own.
<!--/mb-->

### What is assumed rather than measured

The occupancy floor of <!--m:gate.min_occupancy-->15%<!--/m--> is an
**assumption**, not a measurement: it cannot be measured without lane footage, so
it is a documented, configurable number rather than a constant presented as a
finding. So is the <!--m:gate.max_occupancy-->90%<!--/m--> ceiling, the
<!--m:gate.min_structural_change-->0.5<!--/m--> loss of structural agreement over
a <!--m:gate.window-->11<!--/m-->px window that decides whether a window changed,
and the <!--m:gate.min_reference_texture-->1.5<!--/m--> grey levels of local
texture below which the gate declines to answer. Their being assumptions is
stated; presenting one as a measurement would not be.

The gate is shown to reject sensor noise, a flat or blown frame and a
plate-sized object, and to hold an empty lane at `false` across the exposure
range above — all in the test suite, and the parts that need no weights run on
every commit. Its behaviour on **real vehicles at a real lane is NOT MEASURED**
and will stay that way until the physical bench exists.

**Every figure and every generated section in this file is produced by a
command.** Editing one by hand turns the suite red. That rule exists because one
was edited by hand: a measured 0.7% became 0.3% with nothing re-measuring it, the
repository's own test still said 0.7%, and the number passed review by looking
measured. The sections came later, because the rule as originally written
protected the numbers and not the sentences around them.

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
