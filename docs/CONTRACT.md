# The Vehicle ID contract

This is the product's public surface. Everything else in the repository is an
implementation detail that may be rewritten; this is not.

Open Parking AI's own lane controller integrates through exactly what is
described here. There is no private path, no second mode and no in-process
shortcut reserved for us — so if this contract is inadequate, we find out
first.

---

## The READ record

One identification. The unit of output, whether it reaches you synchronously,
by push, or by pull.

```json
{
  "schema_version": 1,
  "read_id": "9f2c1a7d4e8b40c2a1f6d3b8e5c07a91",
  "captured_at": "2026-08-27T14:03:11.482913+00:00",
  "camera_id": "lane-1-entry",
  "identity": {
    "plate": "7ABC123",
    "plate_region": null,
    "make": null,
    "model": null,
    "color": null,
    "marks": []
  },
  "confidence": 0.9962,
  "engine": {
    "name": "openparking-vehicle-id/plates",
    "version": "0.1.0",
    "weights_id": "sha256:4f19c0a7b3e5d281"
  },
  "threshold_applied": 0.99,
  "outcome": "answer",
  "captures_seen": 3,
  "presence": true,
  "presence_confidence": 0.91
}
```

| field | meaning |
|---|---|
| `schema_version` | The record's shape. See compatibility below. |
| `read_id` | Unique per identification, and **stable across re-delivery**. Deduplicate on it. |
| `captured_at` | When the FIRST capture was taken, UTC, ISO 8601 with the offset present. Not when the record was produced, and not when you received it. |
| `camera_id` | Whatever the caller labelled the capture with. The engine does not interpret it. |
| `identity` | What was measured. **Anything not measured is `null`.** |
| `identity.marks` | Distinguishing appearance. Empty means "none were measured", never "the vehicle had none". |
| `confidence` | `[0, 1]`, the engine's own measure for this read. |
| `engine.weights_id` | A digest of the weights actually loaded, not a label somebody typed. Two records that disagree are only worth investigating if you can tell whether the same model produced them. |
| `threshold_applied` | The operating point in force when this record was produced. **Measured for the exact weights named in `weights_id`** — the engine refuses to start on weights whose operating point nobody has measured, rather than stamping a constant onto records a different model produced. |
| `outcome` | `"answer"` or `"fallback"`. There is no third value. |
| `captures_seen` | How many captures produced this record. Present so that one confident record and a batch that disagreed with itself are distinguishable afterwards. |
| `presence` | Whether a **vehicle** was there. `true`, `false`, or `null` for NOT MEASURED. Never folded into `confidence`. |
| `presence_confidence` | How far the measurement sat from the decision boundary. Not a probability, and not claimed to be one. `null` when presence was not measured. |

## Invariants the record itself enforces

These are checked when a record is built and when one is parsed, so a consumer
never has to defend against them:

- `outcome == "answer"` implies `confidence >= threshold_applied`. A record
  cannot claim the engine stood behind it while carrying a confidence the
  stated operating point would have rejected.
- `confidence` and `threshold_applied` are finite numbers within `[0, 1]`.
  Booleans are refused — `true` is not a measurement, and Python would
  otherwise read it as `1.0`.
- `captured_at` parses as ISO 8601 **and carries a UTC offset**. A naive
  timestamp is refused rather than assumed to be UTC.
- `camera_id`, `read_id` and every populated `identity` field are strings.
- `schema_version` is an integer. `true` is not `1` here, whatever Python says.

## Presence is a different question from identity

**"Is a vehicle there" and "can I read its plate" are not the same question, and
the answers have opposite correct responses.**

- `presence: false` — nothing was there. **Do not transact.** No ticket, no
  session, no barrier. This is not a fallback; a fallback means a human deals
  with a vehicle you could not identify, and it ends in a ticket. This ends in
  nothing, because there is no car. Record it, so a lane being worked by
  somebody tripping the loop with a piece of metal shows as a pattern rather
  than as silence.
- `presence: true`, `outcome: "fallback"` — a car is there and could not be
  identified. A filthy, damaged or missing front plate is a **legitimate entry**.
  Admit it to your fallback path. Refusing it is a bug in your integration.
- `presence: null` — **nobody measured it.** Behave exactly as you would have
  before this field existed. Do not read it as `false`: in most languages `null`
  is falsy, and `if (!read.presence)` turns every deployment without a reference
  view into one that refuses every customer.

`false` is the narrow one on purpose. It is the only value that ends a
transaction, so the engine says it only when it has the measurement to back it:
the lane is visible, its ground is still recognisable as the ground the
reference describes, and nothing is standing on it. Note the middle clause — it
is what the measurement actually establishes, and it is weaker than "there is
nothing there". A lane whose ground carries no texture to recognise cannot
support that claim and returns `null` instead. A camera that failed, a view that
no longer matches the reference, a vehicle close enough to fill the frame and a
caller sending plate crops are all `null` — not because nothing is there, but
because nobody can tell, and `null` is the value that means that.

**`null` is not the only thing that happens when the gate loses the lane.** A
scene disturbed enough passes through a band where the gate reads an EMPTY lane
as occupied and answers `true` before it gives up and answers `null`. **What
counts as disturbed enough is not stated here** — see the deployment section
below. That band matters because `true` on an empty lane costs a ticket and an
attendant. Whether it ever becomes a refusal is measured there
too. It is not nothing, and it is not something a reader should have to
discover.

Two invariants the record enforces, so you never have to check them:

- `presence: false` **cannot** carry an identity. Nothing was there to identify;
  a record claiming both does not describe a bad read, it contradicts itself.
- `presence: false` **cannot** be `outcome: "answer"`. There is nothing to stand
  behind.

Presence is additive and does **not** bump `schema_version`. A consumer written
before it existed sees `outcome: "fallback"` for a non-vehicle, which is safe —
it just cannot tell that case apart from an unreadable plate, which is the whole
reason the field exists.

## A batch is one vehicle, and the engine checks

`POST /v1/reads` takes several captures **of one vehicle**. If more than one of
them reads a plate confidently and they disagree, the engine does not pick the
higher score: it answers `fallback`, and `captures_seen` records how many it
was given.

That case is a second vehicle in frame, a tailgater, or a mixed-up buffer, and
the honest answer is that the engine does not know which car is at the barrier.
Picking the strongest read would produce a confident, coherent, in-contract
record naming the wrong car, with nothing anywhere to say so.

`camera_id` on the record is the camera the ANSWER came from, not the first
camera in the batch.

## `fallback` is an answer

The single most important thing to get right when integrating.

A read the engine will not stand behind comes back as a normal, successful
response — HTTP 200, a complete record, `outcome: "fallback"`. It is not an
error, not an empty body, not a 4xx, not a timeout.

- **Do not retry it.** Nothing went wrong. The same captures will produce the
  same answer.
- **Do not treat it as "no vehicle".** Something was there; the engine is not
  sure enough to name it.
- **Do what a human would be asked to do**: fall back. Raise an intercom, take a
  ticket, ask an attendant. That is the behaviour the lane controller
  implements, and it is why a system built on this can be trusted by someone who
  cannot check its work.

`confidence` is still populated on a fallback, and it is still real. It is
reported so you can log and analyse it — not so you can second-guess the
outcome with a threshold of your own. The engine applied a **measured**
operating point; a number you picked is not one.

## Errors, which are a different thing entirely

An error means the engine could not process your request at all:

| | |
|---|---|
| `400` | The request was malformed — bad JSON, no captures, unusable base64, an unsupported content type. Yours to fix; retrying unchanged will fail identically. |
| `413` | The body exceeded the size limit. |
| `404` | No such route, or `GET /v1/reads/last` before anything has been read. |

A read that produced no plate is **not** in this list. That is a `200` with
`outcome: "fallback"`.

## Compatibility

`schema_version` is `1`.

- **Additive changes do not bump it.** New fields may appear. Ignore fields you
  do not recognise rather than rejecting the record.
- **Anything a consumer could notice bumps it** — a field removed, renamed, or
  changed in meaning or type.
- **An unrecognised version is refused, not partially read.** This
  implementation raises rather than guessing which fields still mean what they
  used to, and a consumer in any language should do the same. Half-understanding
  a record about a vehicle is worse than admitting you cannot read it.

`outcome` is a closed set. If a future version needs a third outcome, that is a
version bump, precisely so that no consumer written today can meet a value it
has no branch for.

## Submitting captures

```
POST /v1/reads
Content-Type: application/json

{
  "camera_id": "lane-1-entry",
  "captures": [
    {"image_b64": "...", "captured_at": "2026-08-27T14:03:11+00:00"},
    {"image_b64": "..."}
  ]
}
```

Several captures of **one vehicle**, not several vehicles. The engine takes the
best of the batch — grabbing several frames exists precisely so that one bad
moment, a wiper or a headlight or a bump, does not decide.

For a caller replacing an LPR unit that has exactly one frame and no wish to
base64 it:

```
POST /v1/reads
Content-Type: image/jpeg
X-Camera-Id: lane-1-entry

<raw bytes>
```

Both return the same record shape, wrapped as `{"cursor": N, "read": {...}}`.

### Retrying a submission

Send `request_id` in the body, or an `Idempotency-Key` header. Re-sending the
same key returns the **same record**, and the engine does not run again.

This matters more than it looks. If your request times out while the engine is
still working, the engine finishes, and — with push configured — delivers a
complete `answer` record for that vehicle. Without a key, asking again produces
a second record with a fresh `read_id` for one car, and a consumer doing exactly
what this document says (deduplicate on `read_id`) records two vehicles.

## Pull, and the cursor

`GET /v1/reads?since=N` returns every record with a cursor greater than `N`,
plus the current cursor. The cursor is monotonic within one run of the service
and is **not** durable across a restart: it is a catch-up window for a consumer
that blinked, not a record of anything. The durable copy of a record belongs to
whoever consumes it — and to the push queue until they have it.

If `since` is ahead of the service's own cursor, the response carries
`"reset": true`. That means the service restarted and your saved position no
longer refers to anything. An empty list without that flag would be
indistinguishable from "nothing happened", which is how a consumer silently
misses every read after a restart.

A consumer that needs guaranteed delivery should use push, not polling.

## Push

Configure a URL and every record is POSTed to it as it happens, as
`Content-Type: application/json`, the record at the top level.

- **A record is written to a durable queue before delivery is attempted**, and
  removed only once you have acknowledged it. A process killed between those two
  points re-sends; it does not forget.
- **Duplicates are normal.** `read_id` is stable across attempts. Deduplicate on
  it.
- **Order is preserved.** If one delivery fails, everything behind it waits. A
  consumer that receives an exit before the entry it belongs to prices the stay
  wrong.
- **A `4xx` from you is a refusal, and the record is dropped and counted**, not
  retried forever. Retrying poison blocks everything queued behind it. Anything
  else — a timeout, a `5xx`, a closed socket — is retried.

## Knowing when it is unwell

`GET /v1/health` reports the engine, its version, the weights digest, the
operating point, and — when push is configured — `delivered`, `refused`,
`pending`, `lost`, `damaged` and the last error.

`status` is `"degraded"`, not `"ok"`, when a read was **lost** (it could not
even be written to the queue — a full disk, a permissions change) or when the
queue held a line this build could not read. Those are the two cases where a
record was answered and then existed nowhere, so they are the two cases health
has to be able to say out loud.

A torn line — the queue was being appended to when the power went — is
quarantined to `<queue>.damaged`, counted, and logged. It does not stop the
service starting. A barrier dead until somebody finds a bad line at 2am is a
worse failure than losing the one read that was mid-write.

## Deploying it safely

Three things this release does NOT do, stated rather than left to be discovered:

- **The recogniser has no rejection stage of its own.** It reads text out of
  noise and out of a flat black frame. Confidence filters most of that away, and
  submitting several captures of one vehicle filters most of the rest — a noisy
  feed answers confidently on <!--m:noise.three_capture-->1.4% mean, 0.7-3.3% across 12 seeds x 150 reads<!--/m--> of
  three-capture reads, against <!--m:noise.one_capture-->1.9% mean, 0.0-4.0% across 12 seeds x 150 reads<!--/m--> of
  single-capture ones. **It is not zero.** Measured on weights
  <!--m:noise.weights-->sha256:0de21983b58b0ecd<!--/m-->; a rate without the artefact it
  was measured on is not a figure.

  With a reference view configured, the presence gate stops the read before the
  recogniser sees a frame that carries no picture, and takes those to
  <!--m:noise.gated_three_capture-->0.0% (12 seeds x 150 reads, 1800 in total, none)<!--/m--> and
  <!--m:noise.gated_one_capture-->0.0% (12 seeds x 150 reads, 1800 in total, none)<!--/m--> respectively. **Configure the
  reference.** But read what that covers: it is measured against a *dead feed* —
  sensor noise, a flat or blown frame — which is one of the two ways a camera
  fails. A camera showing a real picture that is simply not a vehicle is not
  closed by it, and neither is a plate hallucinated out of a badly framed but
  perfectly valid scene. **If a wrong answer is expensive for you, keep a second
  gate of your own regardless**: a permit list, a plausibility check, anything
  that is not this engine.
- **Presence is UNVALIDATED against real vehicles, and it is opt-in.** No frame
  of real footage has ever been through this gate; every number about it
  describes drawn rectangles on a drawn lane. It is **off unless you configure a
  reference view** — without one presence is `null` and nothing about your
  integration changes. Switching it on is choosing an unvalidated measurement.

  What it is shown to do: reject sensor noise, a flat or blown frame and a
  plate-sized object, and hold an empty lane at `false` across
  <!--m:exposure.range-->light level 20 to 250 against a reference captured at 90<!--/m-->.
  **How often it admits a lane with a vehicle in it, and what it does with the
  ones it does not, are measured in the section below** — as is what it is shown
  NOT to do, including two bands in which an empty lane reads as OCCUPIED. Read
  that section before you switch this on; the numbers are there and not here,
  because a claim stated in two places goes stale in one of them.

  Real lanes, real cars, real weather: NOT MEASURED until the bench. The
  occupancy floor, the ceiling, the structural threshold and the texture floor
  are assumptions and are configurable for that reason.

- **Presence needs a view of the LANE, not a crop of a plate.** A caller
  submitting tight plate crops has no empty-lane background in frame, so
  occupancy runs to the ceiling and presence is `null` — reads carry on exactly
  as they did before this field existed. That is deliberate: the alternative is
  a `false` that would refuse every customer of every crop-submitting
  deployment.

- **The service is not authenticated.** Anything that can reach the port can
  submit captures and read records, and a consumer cannot tell this engine from
  another process that bound the port first. Keep it on loopback, or on a
  segment you control. Do not expose it.
- **The push queue is trusted local state.** It is a plain file, created `0600`,
  and whatever is in it is delivered. Anything that can write to it can put a
  record of its choosing in front of your consumer. Protect it as you would a
  credential.

## What presence is measured to do, and what it is measured NOT to do

Everything in this section is **generated** from `docs/measured/presence.json`
by `scripts/eval_presence.py --update-docs`, and `tests/test_measured_docs.py`
fails if a word of it drifts from the measurement.

The sentences are generated and not merely the numbers, and that is not
fastidiousness. The previous release stated a correct measured figure — the
gate answers `false` up to 5% streak coverage — and, in the hand-written words
beside it, "and returns `null` above that", which this same evidence file
contradicted: there is a band in which an empty lane reads as OCCUPIED. A true
number lent its credibility to a false sentence, in this document and in the
README, and nothing could see it. So nothing hand-written describes measured
behaviour here any more.

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

**A vehicle is admitted in 106 of the 108 cells**, including at the ground's exact luminance, where the vehicle and the ground it stands on are the same brightness. None of them was refused. In the remaining 2 presence is `null` and the gate reports a camera fault: `reference_not_recognised` in 2. A cell that is not measured is not a refusal — but it is a frame with a car in it that this gate answered nothing about, and the reason it gave names equipment.
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
**Headlights on the floor.** A car with its beams on throws them into frame before the car itself arrives — a large change in the scene caused by a vehicle that is not yet the vehicle. Measured over 8 pools at ambient level 90, with and without the car that cast them. The published figure is the beam's PEAK as a multiple of ambient; the evidence stores what the beam adds, and peak = pool + 1. Both columns are here so the conversion is on the page.

| beam pool, peak x ambient | pool, as stored | ambient level | empty lane (car not yet in frame) | vehicle |
|---|---|---|---|---|
| x1 | 0 | 90 | `false` 0.001 | `true` 0.428 |
| x1.5 | 0.5 | 90 | `false` 0.000 | `true` 0.442 |
| x2 | 1 | 90 | `false` 0.000 | `true` 0.447 |
| x3 | 2 | 90 | `false` 0.001 | `true` 0.453 |
| x4 | 3 | 90 | `true` 0.205 | `true` 0.498 |
| x5 | 4 | 90 | `true` 0.335 | `true` 0.540 |
| x7 | 6 | 90 | `true` 0.477 | `true` 0.639 |
| x9 | 8 | 90 | `true` 0.804 | `true` 0.667 |

At ambient level 90, an empty lane holds at `false` up to a pool of x3 ambient and reads as OCCUPIED from x4 — the beams of a car that has not arrived open a transaction for it.

0 of 8 vehicle scenes were refused. **The model is a limitation of these numbers**: multiplicative pool on a matte floor; no specular glare, no beam cut-off. A gloss or wet floor at night is a specular scene and this is a matte one. NOT MEASURED on any real frame (0 have ever been through this gate).
<!--/mb-->

<!--mb:presence.safety-->
**The one thing that holds everywhere measured.** 0 wrongful refusals in 124 scenes containing a vehicle: 108 matrix cells, 8 weather coverages and 8 headlight pools, each measured with a vehicle in the frame. `false` is the only value that ends a transaction, and no scene measured produced it for a frame with a vehicle in it. Where this gate fails it fails to `null`.

Every one of those 124 scenes is a drawn rectangle on a drawn lane — NOT MEASURED on any real frame (0 have ever been through this gate). The claim is that the measure holds across everything that has been put through it, not that everything has been put through it.
<!--/mb-->

<!--mb:presence.conflation-->
**One reason covers 4 unrelated conditions, and this release cannot tell them apart.** `reference_not_recognised` is reported for all of the following:

- a capture that is not a view of this lane: a camera knocked out of alignment, or a scene rebuilt overnight
- a vehicle close enough to fill the frame
- an ordinary vehicle arriving on low-texture ground under a beam pool
- heavy weather

It is published under `camera_faults` in `GET /v1/health`. That is right for 1 of the 4 — a capture that is not a view of this lane — and wrong for the other 3, where nothing is broken. **Do not read this reason as a confirmed equipment fault** — read it as "the capture no longer matches the reference, for one of several reasons this build cannot separate". Separating them needs a measurement this release does not make, and inventing one would be guessing; naming the conflation is the honest thing available now.

**One of those conditions is a car arriving.** 2 of the 108 separation-matrix cells put an ordinary vehicle — 44% of the frame, not one filling it — in front of the camera and got `reference_not_recognised` back. The gate counts that under `camera_faults`, so an arriving car pages a technician about a working camera. NOT MEASURED on any real frame (0 have ever been through this gate): how often a real entry lands in one of these configurations is not known, and these are drawn rectangles. What is known is that the reason cannot be read as equipment on its own.
<!--/mb-->

---

## Local only

Identification never leaves the device or the LAN. The service binds to
loopback unless configured otherwise, there is no cloud call anywhere in the
path, and the engine works with the internet down. "Integrated" means a local
contract, never a hosted service.

---

Built by 72 Knots Method by 72Knots.ai
