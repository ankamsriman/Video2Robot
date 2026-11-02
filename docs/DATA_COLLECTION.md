# How the data was collected

> **Draft.** Lines marked **[to confirm]** need checking by the person who recorded the videos.

## What was recorded

| Video | Task | Length | Glasses start |
|---|---|---|---|
| `task2_triangle_right_take1` | Three glasses built into a triangle | 29 s | My right side of the table |
| `task3_flip_right_take1` | Triangle, then a fourth glass flipped and placed opening up | 45 s | My right side of the table |
| `old_left_side/task2_triangle_left_take1` | Triangle | 25 s | My left side |
| `old_left_side/task3_flip_left_take1` | Flipped tower | 32 s | My left side |

Samples of the first two are in `data/raw/sample/`. They are compressed `.mp4` copies of the originals, at full size.

## Setup

- **Files:** `.mov`, 1620 x 1080, 30 frames per second.
- **Camera:** fixed in place, beyond the far edge of the table, looking down at it and at me. Measured movement during each take was under 1 pixel. **[to confirm: phone model, and what held it]**
- **Table:** dark striped top, measured at 80 x 50 cm, to about 1 cm.
- **Glasses:** steel, all alike, 11.5 cm tall, 7.3 cm across the rim, 5.5 cm across the closed end. Measured with an on-screen ruler app. **[to confirm: tool used]**
- **Light:** indoor room. **[to confirm: which lights, and the time of day]**
- **Marker:** none. The table's corners act as the reference.

## How each take was done

1. Stand at the table's near edge, facing the camera, with the glasses upside down, closed end up.
2. Move one glass at a time with my right hand, holding it from the side so its top stays visible.
3. Let go and move my hand clear between glasses.
4. For Task 3, turn the last glass over in the air and stand it on top, opening up.

## Labels

There are no hand-written labels of the task. The only hand-made input is four rough table corners per video, in `data/labels/<video>_table.json`.

## Problems

- The camera's automatic exposure changes the brightness during a take, which is why the system compares glass tops with the table in the same frame.
- The first two takes started the glasses on my left, out of the robot's reach, so I recorded again on my right.
- My face is out of frame, but my body and feet appear in the videos.
