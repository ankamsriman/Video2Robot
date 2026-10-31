# Video2Robot Task Specification & Implementation Tracker

Based on the **SeeDo** framework (*"VLM See, Robot Do: Human Demo Video to Robot Action Plan via Vision Language Model"*, NYU, arXiv:2410.08792) and expanding upon preliminary tracking explorations in `Demo2Robot`.

---

## Project Goal
Transform unstructured human demonstration videos into robot-executable symbolic plans and continuous control trajectories in MuJoCo simulation via:
1. Kinematic Keyframe Selection (MediaPipe hand velocity valleys)
2. Visual Prompting & Open-Vocabulary Tracking (Grounding DINO + SAM 2)
3. Chained Chain-of-Thought (CoT) VLM Reasoning (Gemini / GPT-4o)
4. Parameterized Action Primitives & Execution in MuJoCo (6-DOF robot arm)
5. Automated Benchmark Suite (TSR, FSR, SSR, error diagnosis)

---

## Phase-Wise Checklist

### [ ] Phase 1: Environment Setup & Project Scaffolding
- [ ] Create Python virtual environment and configure dependencies (`opencv-python`, `mediapipe`, `torch`, `torchvision`, `mujoco`, `scipy`, `pydantic`).
- [ ] Establish directory hierarchy (`configs/`, `data/`, `src/`, `tests/`).
- [ ] Link / copy demonstration videos from `../Demo2Robot/data/raw/` to `data/raw/`.
- [ ] Create baseline configuration file `configs/default.yaml`.

### [ ] Phase 2: Keyframe Selection via Hand Kinematics
- [ ] Implement `src/keyframe_selector/hand_detector.py` using MediaPipe Hands.
- [ ] Implement `src/keyframe_selector/speed_analyzer.py` to calculate hand center speed profile $v(t) = \|\mathbf{x}_t - \mathbf{x}_{t-1}\| / \Delta t$.
- [ ] Implement Gaussian / Savitzky-Golay smoothing and local minima (valley) detector.
- [ ] Prune boundary frames and cluster adjacent candidate detections.
- [ ] Implement visualization tool saving velocity curves with marked pick/place keyframe timestamps.

### [ ] Phase 3: Visual Prompting & SAM 2 Video Tracking
- [ ] Implement `src/perception/object_detector.py` for initial frame object discovery (Prompt 1) and open-vocabulary bounding box detection (Grounding DINO / Florence-2).
- [ ] Implement `src/perception/sam2_tracker.py` using SAM 2 to track instances across keyframes with persistent IDs.
- [ ] Implement visual prompt overlay renderer (colored mask contours, tracking ID labels, and centroid coordinates $(u, v)$).
- [ ] Export visual-prompted keyframe sequences to `data/processed/visual_prompts/`.

### [ ] Phase 4: Chained CoT VLM Reasoning & Task Plan Generation
- [ ] Implement `src/reasoning/prompts.py` containing standard SeeDo prompts (Filtering, Pick detection, Reference detection, Spatial relation classification).
- [ ] Implement `src/reasoning/vlm_planner.py` interacting with VLM (Gemini 1.5 Pro / GPT-4o / open-source VLM).
- [ ] Pair keyframes into sequential $(\text{Pick}, \text{Place})$ intervals.
- [ ] Classify 6 canonical spatial relations: `in`, `on top of`, `at the back of`, `in front of`, `to the left`, `to the right`.
- [ ] Output validated, structured JSON / Pydantic plans.

### [ ] Phase 5: MuJoCo Robot Simulation & Code-as-Policies Execution
- [ ] Create 6-DOF robot simulation scene in MuJoCo (UR5e / Franka Panda with gripper on workspace table).
- [ ] Implement differential IK solver and smooth Cartesian trajectory interpolator in `src/simulation/robot_env.py`.
- [ ] Implement parameterized action primitives in `src/simulation/primitives.py`:
  - `pick(object_id)`
  - `place_relative(object_id, ref_id, relation)`
- [ ] Map 2D visual coordinate offsets to 3D Cartesian workspace offsets.
- [ ] Build automated simulation video recorder saving execution rollouts.

### [ ] Phase 6: Automated Evaluation Suite & Error Diagnosis
- [ ] Implement `src/evaluation/metrics.py`:
  - **TSR (Task Success Rate)**: Strict sequence equality check.
  - **FSR (Final State Success Rate)**: Final relational state graph alignment.
  - **SSR (Step Success Rate)**: Two-pointer sequence alignment score.
- [ ] Implement automated error categorization:
  - Vision Error (object recognition mismatch).
  - Spatial Error (incorrect spatial relation).
  - Temporal Error (wrong order, omitted or redundant steps).
- [ ] Implement benchmarking CLI script outputting Markdown and CSV evaluation reports.

### [ ] Phase 7: CLI Pipeline & Interactive Dashboard
- [ ] Implement `main.py` CLI supporting modular execution (`--step keyframes`, `--step perception`, `--step plan`, `--step sim`, `--step full`).
- [ ] Build interactive visualizer (Streamlit / Gradio) displaying demo video, hand velocity graph, prompt-annotated keyframes, VLM reasoning traces, and MuJoCo execution playback.
