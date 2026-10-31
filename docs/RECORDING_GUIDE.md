# Recording guide

This is how to film the demo videos for Phase 2. The layout matches what the simulated G1 can actually reach, so please keep to these measurements.

## What you need
- A phone on a tripod
- A clear table, at least 60 cm x 60 cm
- 3 objects (for example fruit or small boxes)
- 2 low open containers (for example shallow trays or box lids)
- The printed marker: `docs/marker.pdf`, printed at 100% scale, not "fit to page"
- A ruler and some masking tape

## Choosing objects and containers

| Item | Size | Why |
|---|---|---|
| Objects | 6-8 cm side to side, 5-8 cm tall | The G1's fingers sit about 5.8 cm apart, so narrower objects pivot in its grip |
| Containers | About 15 x 18 cm, under 5 cm tall | Fits the robot's reach and leaves room for two objects |

Choose objects that look clearly different, ideally in different colours, with the table a contrasting colour. This makes tracking much easier.

## Table layout
Stand on the **robot side** of the table, where the robot would stand. Mark this area with tape:

```
                        far side of the table
      +--------------------------------------------------+
      |    [ container A ]            [ container B ]    |   tray centres: about 26 cm past the marker
      |                                                  |
      |      (object)      (object)      (object)        |   objects: about 11 cm past the marker
      |                                                  |
      |                     [MARKER]                     |   marker centre
      +--------------------------------------------------+
                  robot side (near edge of the table)
```

- **Marker:** flat on the table, about 8 cm from the near edge, with its arrow pointing away from you. Tape it down so it can't move.
- **Objects:** in a row about 11 cm beyond the marker, spaced about 12 cm apart.
- **Containers:** about 26 cm beyond the marker, side by side, with centres about 24 cm apart.

Small changes in position between videos are fine and useful.

## Camera setup
- Put the phone on a tripod facing the table from **the far side or a corner**, looking down at about 45-60 degrees.
- Make sure the marker, all objects and both containers stay in frame for the whole video.
- Use 1080p at 30 frames per second, in landscape.
- Lock focus and exposure. On iPhone, press and hold on the table until "AE/AF Lock" appears.
- Use bright, even light and avoid strong shadows.
- **Do not move the phone between videos.** If it gets bumped, tell me, because the positions are measured from this view.

## How to perform the demo
1. Start with your hands out of frame, then count to 2.
2. Use **one hand**. Reach in, pick up one object, and place it.
3. **Pause briefly** at each pick and each place. The software finds these moments from your hand slowing down.
4. Try not to cover the object with your hand more than you need to.
5. Move your hand out of frame between steps if you can, then finish with your hands out of frame for 2 seconds.

Each video should be about 20-40 seconds.

## What to record
Record **about 10 videos**. Vary the order and which object goes where, and put two objects in one container in some of them.

Save them as `data/raw/video_01.mp4`, `data/raw/video_02.mp4`, and so on. Three of these will be kept aside for the final test and won't be used for tuning.

## Labelling each video
For each video, create a matching file such as `data/labels/video_01.json`. Times are in seconds, and rough values are fine:

```json
{
  "moments": [
    { "kind": "pick", "time": 3.2 },
    { "kind": "place", "time": 5.8 }
  ],
  "plan": {
    "steps": [
      { "object": "apple", "reference": "tray_a", "relation": "in" }
    ]
  }
}
```

`relation` can be `in`, `on_top_of`, `at_the_back_of`, `in_front_of`, `to_the_left` or `to_the_right`. Sorting tasks will mostly use `in`.

## Measurements to send me
- Each object's width, depth and height, and its name
- Each container's width, depth and height, and its name
- The printed marker's measured side length, to confirm it printed at the right scale

I'll update the digital twin with these sizes.
