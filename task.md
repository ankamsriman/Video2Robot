​Awign and Armature AI Labs in partnership with RIC (Robotics India Community) and Tanna TechBiz (NVIDIA Preferred Partner) are hiring interns to work on humanoid robot learning. We’re skipping the resume screen and asking you to build something instead.

​Edit: Upon approval, please check your email with whatsapp link and join the group. All updates will be shared there first.

​Description

​Record yourself doing a simple task on your phone. Use that footage to make a humanoid robot do it in simulation. The strongest submissions get a 2-month, full-time robot learning internship and full-time roles with Awign hardware team in Bangalore.

​The challenge

​Collect a small manipulation dataset yourself. A phone camera is enough, and egocentric footage works well. Then use it to drive a humanoid or dexterous robot hand in simulation. Unitree G1 and MuJoCo are preferred, and alternatives are welcome.

​There’s one hard rule: the data you personally recorded has to play a real role in your approach. Everything else is open.

​Here’s an example: Film your hand picking up a cup and placing it on a coaster. Retarget your hand poses to the G1’s five-finger hand. Post-train a small VLA like SmolVLA on that data. Then show the G1 doing the task in MuJoCo, using fingertip contact to confirm the grasp.

​Directions you could take

​Post-train a policy on a simple task using your own recordings

​Retarget human hand motion to a five-finger hand, and use tactile signals for grasp detection or force control

​Bootstrap a policy and push its success rate up with RL

​Take the world-model route: show video or state prediction, with less focus on policy success

​Take a standard policy and make it run much faster than the baseline

​Pick your robot

​The Unitree G1 is our default, but we’d love to see other embodiments. Retargeting your data well to a less common body counts in your favour. All of these have free MuJoCo models in MuJoCo Menagerie:

​Unitree G1 (default): a 1.3 m humanoid. Pair it with a five-finger hand (Inspire, Dex5, or similar).

​Booster T1: a smaller, lighter humanoid, good for fast RL iterations.

​Apptronik Apollo: a full-size, human-scale humanoid. Human data maps more directly here.

​Pick your simulator

​MuJoCo is preferred. Use whatever runs on the hardware you have:

​MuJoCo (preferred): a pip install that runs on a laptop CPU or free Colab. Use MJX or MuJoCo Playground for GPU-parallel RL.

​PyBullet: the lightest option, and it runs anywhere.

​Isaac Sim / Isaac Lab: the best for large-scale RL and photoreal cameras. Needs an NVIDIA RTX GPU.

​LIBERO / robosuite: tabletop benchmarks built on MuJoCo, good for starting with an arm.

​What to submit

​A public GitHub repository by Wed 4 Nov 2026, 23:59 IST, containing:

​A README or short deck: how to run it, example outputs (videos or GIFs), your design choices, and what worked and what didn’t

​Your recorded data (or a representative sample), plus a note on how you collected it

​We’ll email the submission form to approved registrants.

​How we judge

​Creativity: as long as your own data genuinely matters to the result

​Results: how well the robot performs in simulation, and/or the quality of your world-model predictions

​Clarity: a simple implementation and an honest, clear presentation. No AI slop please.

​Challenge Winners:

​Paid Internships with Awign : Full-time, 6 days a week, in Bangalore, for 2 months. You’ll work daily with the founding team and engineers working on humanoid robots.

​Strong performers will be considered for full-time roles with the Awign hardware team.

About

​Awign (a Mynavi company): Robotics data collection at scale

​Armature AI Labs: a shared physical AI and robotics lab in HSR Layout, Bangalore.

​RIC: Robotics India Community, democratising robotics in India

​TTB (Tanna TechBiz): Physical AI Solutions, NVIDIA preferred partner