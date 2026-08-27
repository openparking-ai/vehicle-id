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
  noise and out of a flat black frame. The presence gate is what closes that —
  measured, a noisy feed goes from 2.3% confident answers to 0.0% with the gate
  in front — but the gate needs a reference view of the empty lane, and without
  one presence is `null` and the old behaviour applies. **Configure the
  reference.** And if a wrong answer is expensive for you, keep a second gate of
  your own regardless: a permit list, a plausibility check, anything that is not
  this engine.
- **The gate has not been measured on real vehicles.** It is shown to reject
  noise, a plate-sized object and scattered speckle, and to admit a readable
  plate. Real lanes, real cars, real weather: NOT MEASURED until the bench. The
  occupancy threshold is an assumption and is configurable for that reason.

- **The service is not authenticated.** Anything that can reach the port can
  submit captures and read records, and a consumer cannot tell this engine from
  another process that bound the port first. Keep it on loopback, or on a
  segment you control. Do not expose it.
- **The push queue is trusted local state.** It is a plain file, created `0600`,
  and whatever is in it is delivered. Anything that can write to it can put a
  record of its choosing in front of your consumer. Protect it as you would a
  credential.

## Local only

Identification never leaves the device or the LAN. The service binds to
loopback unless configured otherwise, there is no cloud call anywhere in the
path, and the engine works with the internet down. "Integrated" means a local
contract, never a hosted service.

---

Built by 72 Knots Method by 72Knots.ai
