# Recording guide

**The current tasks use steel glasses, with no marker.** Follow "Steel glasses" below. The rest of this guide is for the earlier block-sorting pipeline, which needs a printed marker.

## Steel glasses (Task 2 and Task 3)

Keep the same camera and table position during a take, with all four table corners and the whole tower in view. Keep your hands off the table for 2 seconds at the start and end. Hold each glass from the side, around the middle, so its bright top stays visible, and move your hand clear after letting go.

Everything below is measured to where each glass's **base** stands: from your edge of the table, and sideways from its centre line.

| Task | Stage | Area | Picture |
|---|---|---|---|
| **Task 2** | Glasses start | 12-34 cm from your edge, from the centre line to 30 cm on your right | `recording_views/task2_initial_area.png` |
| **Task 2** | Tower middle ends | 24-36 cm from your edge, 12 cm on your left to 24 cm on your right | `recording_views/task2_final_area.png` |
| **Task 3** | Glasses 1, 2, 3 start | 13-19 cm from your edge, 14 cm on your left to 10 cm on your right | `recording_views/task3_initial_area.png` |
| **Task 3** | Glass to flip starts | 23-28 cm from your edge, 4 cm on your left to 10 cm on your right | `recording_views/task3_initial_area.png` |
| **Task 3** | Tower middle ends | 20-27 cm from your edge, 26-31 cm on your right | `recording_views/task3_final_area.png` |

Keep glasses at least 12 cm apart, centre to centre, and at least 15 cm from where the tower will stand. In the phone's view, left and right are swapped, because the phone faces you.

Name takes like `data/raw/task2_triangle_right_take2.mov`, and add `data/labels/<name>_table.json` with four rough table corners: near-left, near-right, far-right and far-left as they appear in the picture.

# Block sorting (earlier work)

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

## What the phone should see
Pictures of the exact view for each task are in `docs/recording_views/`. They're drawn in the simulator to real measurements, and they aren't photos. `task1_start.png` and `task1_end.png` show Task 1, and Tasks 2 and 3 have the same pair.

## What of you should be in the video
Pictures are in `docs/recording_views/`, named `person_1_start.png` to `person_4_mistake.png`. They use a simple simulator model of a person, not a photo.

- **In the shot:** your hand and forearm while you work, and possibly your waist at the top edge. Your head and face aren't needed, so keep them out, since this data will be public.
- **Your hand:** one bare hand with all five fingers showing, no gloves. Hand detection needs to see the fingers.
- **Your grip:** for blocks, from above or the sides, so the phone can still see the object's front. For steel glasses, always from the side around the middle, because glasses are followed by their bright closed tops, so those must stay uncovered.
- **Your reach:** come from your side of the table, and never reach past the trays to the marker.
- **Your clothes:** a plain shirt, not the same colour as any object or tray, because objects are followed by colour. Don't use a tray close to your skin colour.
- **Nobody else in the shot.**

## Table layout
Stand on the **robot side** of the table, where the robot would stand. Measure from that near edge, and mark the spots with tape:

```
                 far side of the table, where the phone is
      +--------------------------------------------------+
      |                     [MARKER]                     |   marker centre: 48 cm from your edge
      |                                                  |
      |    [ container A ]            [ container B ]    |   tray centres: 34 cm from your edge
      |                                                  |
      |      (object)      (object)      (object)        |   objects: 19 cm from your edge
      |                                                  |
      +--------------------------------------------------+
                  robot side (near edge of the table)
```

- **Marker:** flat on the table, with its arrow pointing away from you, towards the phone. It sits at the phone's end so objects can't block its view. Cut the page down, leaving about 3 cm of white around the black square, and draw the arrow on before cutting if it would be cut off. Tape it down so it can't move.
- **Objects:** measured from a line running straight towards you from the marker's centre: 18 cm and 6 cm to your right, and 6 cm to your left.
- **Containers:** 11 cm to your right of that line, and 13 cm to your left.

Small changes in position between videos are fine and useful.

## Camera setup
- Put the phone on a tripod facing the table from **the far side**, about 75 cm above the tabletop and about 30 cm beyond the table's far edge, looking down at about 50 degrees. A short tripod works if it stands on a second table of about the same height.
- Make sure the marker, all objects and both containers stay in frame for the whole video.
- Check that **nothing covers the marker**, even partly, including your hand during the first few seconds.
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

## Check your first video
Record one test video, then run this. It reports what to fix before you film the rest:

```bash
PYTHONPATH=. .venv/bin/python scripts/check_video.py --video data/raw/video_01.mp4
```

It needs your objects' colours, which you can read from the video by pointing at each object's pixel position. Or send me the video and I'll set them up.

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

## Optional: the cup triangle
Only if we decide to show this task. Use 3 opaque cups, each a different colour, turned upside down, about 7-8 cm wide at the rim.

- **Start:** cups apart, within the same area as the objects above.
- **Build:** place two side by side, touching, then the third across their tops. Place them in the same order each time, if you can.
- **Camera:** from the far side, so the two lower cups appear **side by side** in the picture. From above or at an angle, the top cup hides a lower one and tracking fails.
- **Marker:** make sure it's fully visible at the start of the video, before cups cover it.

Record 3-4 videos, saved as `data/raw/cups_01.mp4`, and so on.

## Measurements to send me
- Each object's width, depth and height, and its name
- Each container's width, depth and height, and its name
- The printed marker's measured side length, to confirm it printed at the right scale

I'll update the digital twin with these sizes.
