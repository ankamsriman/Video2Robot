# How the data was collected

> One line marked **[to confirm]** still needs checking by the person who recorded the videos.

## What was recorded

| Video | Task | Length | Glasses start |
|---|---|---|---|
| `task2_triangle_right_take1` | Three glasses built into a triangle | 29 s | My right side of the table |
| `task3_flip_right_take1` | Triangle, then a fourth glass flipped and placed opening up | 45 s | My right side of the table |
| `task2_triangle_take2` to `take8` | Triangle, seven takes in different layouts, camera moved slightly from take 1 | 17-24 s | Various, including both sides |
| `task2_triangle_check` | Triangle. **Excluded** from later tests by me: its first glass stood beyond the robot's reach | 21 s | My left side |
| `task3_flip_take2` to `take7` | Flipped tower, six takes in different layouts, same camera as the Task 2 takes above | 23-34 s | Various, including both sides |
| `task3_flip_take8`, `take9` | Flipped tower, in a layout chosen to suit the robot | 29-30 s | My right side of the table |
| `old_left_side/task2_triangle_left_take1` | Triangle | 25 s | My left side |
| `old_left_side/task3_flip_left_take1` | Flipped tower | 32 s | My left side |
| `task2_triangle_clutter_take1`, `take2` (4 Nov) | Triangle on a cluttered table: phone, earbud case, card, charger, tube, plug, box, steel laptop stand | 24 s | My right side |
| `task2_triangle_hard_take1` to `take4` (4 Nov) | Triangle, made hard on purpose: a 4th glass never moved (1), a glass the right way up (2), the steel laptop stand (3), a plug moved mid-video (4) | 20-29 s | Various |
| `task2_triangle_sunlight_take1` (4 Nov) | Triangle on a cluttered table, lit by sunlight only | 21 s | Both sides |

Samples of the first two, and of `task2_triangle_take5`, `task2_triangle_take6` and `task3_flip_take9`, are in `data/raw/sample/`. They are compressed `.mp4` copies of the originals, at full size.

## Setup

- **Files:** `.mov`, 1620 x 1080, 30 frames per second.
- **Camera:** a Logitech Brio 100 webcam, fixed in place beyond the far edge of the table, looking down at it and at me. Worked out by the code from the table's corners: about 97 cm above the table, 104 cm from its centre, looking down about 43 degrees, the same in every video. Measured movement during each take was under 1 pixel.
- **Table:** dark striped top, measured at 80 x 50 cm, to about 1 cm.
- **Glasses:** steel, all alike, 11.5 cm tall, 7.3 cm across the rim, 5.5 cm across the closed end. **[to confirm: measured with an on-screen ruler app?]**
- **Light:** indoor room lit by the floor's LED lights, for every video except `task2_triangle_sunlight_take1`, which was lit by sunlight only.
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
