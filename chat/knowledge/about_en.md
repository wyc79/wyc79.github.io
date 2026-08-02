# knowledge/about_en.md — gate-calibration corpus (feeds retrieval AND the en gate)

**This IS gate-calibration data, not just supplementary retrieval text.** The
English gate is wired to this file: it scores against this file's curated
`## Heading` sections (currently 55), not against raw page chunks the way an
earlier version of this project did. Edit accordingly: a section reworded for
retrieval phrasing can move or erase the gate's separation margin. Unlike the
Chinese gate (below), there is no silent-disable fallback for English —
`index_builder.py`'s `_check_en_gate_margin` **raises and aborts the build**
if a rebuild's calibrated margin goes negative, so a bad edit is caught at
build time, not discovered later as a live false-refusal bug. Still rebuild
and read the build log's gate line after every edit; an aborted build means
this file needs another look before the index can be regenerated at all.

**The lesson from the Chinese file, and why the English gate's build now
raises instead of warning:** commit 10be374 reworded about_zh.md by a single
line, with no sections added or removed, and dropped its calibration margin
from healthy to near zero. The zh gate has been disabled ever since (still
the case today) — undetected by any test at the time, because the corpus
still read fine; only the calibration number showed the damage. That failure
mode is exactly what the English gate's build-time guard exists to prevent
for this file: silent margin decay is no longer possible for English, because
the build itself refuses to ship one.

Sections exist because visitors ask in hiring vocabulary the pages
themselves never use — resume, CV, background, qualifications. Keep every
fact consistent with the actual site pages. **Rebuild after editing**
(`python scripts/build_index.py`) and check the build log's gate line for the
current threshold/margin — don't trust a number quoted here or elsewhere, it
moves on every rebuild of this file.

Naming: both languages live in this one folder split by suffix — about_en.md
(this file, English, the en-gate corpus) and about_zh.md (Chinese; ALSO a gate
corpus by design). `load_knowledge(dir, lang)` reads only `*_<lang>.md`, so the
en gate never ingests zh chunks and vice-versa.

Whether a Chinese gate actually ships is decided per build and is deliberately
NOT stated here: `build_index.py` writes `data/gate_zh_bge.json` only when that
build's zh calibration separates on-/off-topic, and removes a stale one only
when this build's own calibration ran and failed. This header used to assert an
answer instead, and was wrong — the same class of claim in the same place was
flagged by two consecutive whole-branch reviews. For the current state read the
build log's `zh gate:` line, or `eval/README.md`'s Known Limits, which is kept
current. When no zh gate ships, Chinese questions bypass the local gate via
`cjk_bypass` and are guarded by the LLM system prompt instead.

## Who this portfolio site is about
link: index.html
The site belongs to Yuanchen Wang, credited in the page header as a game
developer and a computer science master's student at USC. His work spans
combat design, gameplay programming, and engine-level systems, and this chat
widget is one of his own projects.

## A short summary of his background
link: index.html
The page's own description calls him an aspiring game developer with a
background in psychology, neuroscience, and biomedical informatics, now
pursuing a computer science master's in game development.

## From brain science into game development
link: pages/education.html
He began with undergraduate degrees in brain and cognitive science and
psychology, went on to a biomedical informatics master's at Harvard Medical
School, and only afterward started a computer science master's focused on
games at USC.

## A starting point for browsing his portfolio
link: pages/projects.html
The projects page separates his shipped games from other related
engineering work; among the shipped titles, Cemented Dreams carries the
deepest combat and level design responsibility and is a reasonable first
stop.

## How to get in touch
link: index.html
The landing page header carries direct contact links next to his name and
logo: an email address, a LinkedIn profile, and a GitHub account.

## His first fully solo-built game
link: pages/gyrotris.html
Gyrotris was made entirely by him for Untitled Game Jam #100: he alone
handled the design, the GDScript programming in Godot 4.3, and the pixel
art assets, made in Piskel. It was his first complete shipped title.

## Small interactive tools built into the site
link: pages/toolbox.html
The toolbox page hosts two utilities built directly into the site rather
than linking out: a word cloud generator with adjustable word count and
scaling, and a QR code generator with adjustable output size.

## Degrees completed versus degrees in progress
link: pages/education.html
He holds two completed bachelor's degrees from the University of
Rochester — a B.S. in Brain & Cognitive Science and a B.A. in Psychology —
plus a completed master's degree from Harvard Medical School, and is
currently partway through a second, ongoing master's degree at USC.

## Languages he speaks
link: pages/skills.html
His skills page lists Chinese as a native language, English at a proficient
level, and elementary Japanese and German.

## His research career before game development
link: pages/publications.html
Before working in games he was involved in neuroscience, psychology, and
biomedical informatics research, contributing to peer-reviewed studies on
brain function.

## His current program at USC
link: pages/education.html
He is enrolled in USC's Master of Science in Computer Science, Game
Development track, running from August 2025 through May 2027, with
coursework in 3-D graphics and rendering and in game engine development.

## Project roles beyond designer and programmer
link: pages/projects.html
His project credits also include CAD Modeler on the Aegis Sword replica,
Python Developer and QA Lead on the automatic differentiation toolbox, and
Producer on a studio game prototype, alongside his design and engineering
roles.

## The project where he holds the broadest design role
link: pages/cemented-dreams.html
Cemented Dreams is where he holds the most design responsibility at once:
he is credited as Combat Designer, Gameplay Engineer, and Level Designer on
the same third-person action game.

## How the projects page is organized
link: pages/projects.html
The projects page is split into a games section (Cemented Dreams, Nothing
Can Go Wrong, Code Breaker, Gyrotris) and an "other related works" section
covering the chat agent, Prime Engine, physical game prototypes, the 3D
rendering project, Aegis Sword, and the automatic differentiation toolbox.

## Shipped games versus supporting engineering work
link: pages/projects.html
Not every project on the site is a full game: Prime Engine, the 3D
rendering project, and the automatic differentiation toolbox are
engineering-only pieces built as coursework or standalone systems rather
than complete playable titles.

## What the chat widget itself demonstrates about him
link: index.html
The chat assistant embedded on this site was designed and built by him end
to end, so it doubles as a working sample of his applied AI and full-stack
engineering rather than only a feature for visitors.

## Combined design and engineering responsibilities on one project
link: pages/cemented-dreams.html
On Cemented Dreams he simultaneously held the Combat Designer, Gameplay
Engineer, and Level Designer roles, building the core systems for the
game's combat and player movement in Unreal Engine 5.

## Grapple traversal as a core movement mechanic
link: pages/cemented-dreams.html
He designed and implemented grapple traversal as a core movement mechanic
in Cemented Dreams, paired with sliding movement built to keep combat
responsive within tight indoor environments.

## Aim-assist as a combat gameplay feature
link: pages/cemented-dreams.html
Among the gameplay behaviors he built for Cemented Dreams is aim-assist,
wired through C++ gameplay components with a Blueprint interface for the
rest of the combat systems.

## Combat parameters exposed for designer tuning
link: pages/cemented-dreams.html
On Cemented Dreams he exposed key combat parameters through Blueprint so
movement feel and combat pacing could be adjusted during design iteration
without rewriting the underlying C++.

## Level design around a growing Hive structure
link: pages/cemented-dreams.html
He designed the Hive level in Cemented Dreams, where a mechanical growth
spreading through the building tears open new entrances and forms
traversal paths and combat arenas as it expands.

## Iterating on prototypes through repeated playtests
link: pages/game-design-workshop.html
In CTIN 488, USC's Game Design Workshop, he iterated on tabletop prototypes
across weekly playtests, rewriting rules and pacing based on what confused
players at the table.

## Simulation-backed balance tuning on a final project
link: pages/game-design-workshop.html
For the CTIN 488 final project, a social-hierarchy survival game, the team
ran a balance pass supported by 200,000-trial simulations before adjusting
the deck composition and win conditions.

## Two studio prototypes chosen for a class showcase
link: pages/game-design-workshop.html
Two of his three CTIN 488 studio projects, the Up the River variation and
the bird-courtship bidding game Too Bird To Handle, were selected for
in-class presentation.

## Serving as producer on a studio team
link: pages/game-design-workshop.html
His listed role on the CTIN 488 final project was Producer; the team
credits him with helping lock the core loop, driving rulebook cleanup,
running the simulation-backed balance checks, and supporting the final
slide-deck and presentation-video prep, alongside his teammates.

## Combat systems engineering on Code Breaker
link: pages/codebreaker.html
On Code Breaker he was the primary gameplay engineer, implementing the core
combat systems and gameplay architecture for a Unity title centered on
structured player decision-making.

## Coding his own gameplay prototypes
link: pages/skills.html
He is not limited to paper design: his toolset includes C++, C#, and
Blueprint, and Cemented Dreams shows him writing the underlying gameplay
code himself rather than handing it to another engineer.

## Removing off-screen geometry before rendering
link: pages/prime-engine.html
On Prime Engine he implemented view frustum culling: each mesh's
world-space bounding box is tested against the camera's six frustum planes
so objects outside the view are skipped before rendering.

## A bounding volume hierarchy for large scenes
link: pages/prime-engine.html
He extended frustum culling into a bounding volume hierarchy on Prime
Engine, with a static tree built once via median splits for stationary
geometry, and a dynamic tree rebuilt every frame from Morton-coded object
positions for moving objects.

## Collision response that keeps momentum along a surface
link: pages/prime-engine.html
The physics he built for Prime Engine removes only the blocked component
of a movement vector on contact, so the remaining motion keeps the object
moving along whatever surface it struck rather than halting outright.

## Layered animation states in a custom engine
link: pages/prime-engine.html
He extended Prime Engine's animation state machine with full-body blends,
partial-body overrides limited to a joint range, and additive layers
stacked on an existing pose, along with debug visualization to inspect the
results.

## Hand-written HLSL for a lava shader effect
link: pages/3d-rendering.html
For the 3D rendering project he hand-wrote a lava-and-cracks terrain shader
in HLSL, using 3D Worley noise for the crack boundaries and a
third-nearest-distance metric to round the crack junctions.

## A real-time rendering demo, not an offline render
link: pages/3d-rendering.html
His 3D rendering project runs live inside Unity's Universal Render
Pipeline: a procedural asteroid strikes terrain and a shockwave expands
outward, revealing molten cracks as it moves, all computed in real time.

## C++ gameplay code underneath the Blueprint layer
link: pages/cemented-dreams.html
Cemented Dreams' gameplay systems were built from Unreal Engine C++ code
first, with Blueprint layered on top for rapid iteration, rather than being
assembled purely in Blueprint.

## Which engine powers Cemented Dreams
link: pages/cemented-dreams.html
Cemented Dreams was built in Unreal Engine 5, using a combination of C++
gameplay code and Blueprint for its combat mechanics and player movement.

## A tally of the engines he has used
link: pages/skills.html
His game-engine experience spans three engines: Unreal Engine 5, Unity, and
Godot, with C# and C++ as his main gameplay-programming languages across
them.

## Languages he codes in outside of engines
link: pages/skills.html
His listed programming languages, outside what he uses inside engines,
include Python, R, C, MATLAB, Java, and SQL, in addition to C++ and C#.

## Formal algorithms coursework at Rochester
link: pages/education.html
His University of Rochester degrees (Aug 2017 – May 2021) carried a minor
in computer science, with coursework spanning Discrete Mathematics, Data
Structures & Algorithms, and Computation & Formal Systems — a theory
foundation underneath his later hands-on engine work.

## Courses touching AI and statistics
link: pages/education.html
He took an Artificial Intelligence course as an undergraduate at Rochester
and a Machine Learning course during his Harvard biomedical informatics
master's, in addition to his rendering and engine coursework at USC.

## Server-side retrieval and judgment for the chat widget
link: pages/chat-agent.html
For this site's chat widget, nearest-neighbor search over site content
happens entirely inside a small server function, against its own bundled
retrieval corpus, not in the visitor's browser; the browser only ever
sends the question, and the query embedding and relevance-gate check
happen through that same function too.

## Separate gate models per language
link: pages/chat-agent.html
The chat widget runs a different relevance-gate model for each language:
an English MiniLM model and a quantized bge-small-zh model for Chinese,
each calibrated with its own threshold.

## A fallback when the chat server is down
link: pages/chat-agent.html
If the chat widget's backend function cannot be reached, it drops to a
consent-gated model running inside the browser that performs retrieval
locally instead of the feature simply failing.

## Resisting name-drop prompt injection
link: pages/chat-agent.html
The chat widget's gate collapses any mention of his name to one token
before judging the rest of a message, so an instruction that only invokes
his name to smuggle in an unrelated request still gets stripped down and
refused.

## Per-turn logs for auditing conversations
link: pages/chat-agent.html
Each turn through the chat widget carries a request ID and a hashed
session ID, and a passing turn's log includes the gate outcome, the
retrieved passages, and the model's input and output.

## Content pipeline behind the chat widget
link: index.html
The chat agent sits on top of a tested Python pipeline that chunks,
embeds, and indexes site content, rather than being a single hand-tuned
script wired directly to a language model.

## Tailoring answers to the visitor's role
link: index.html
The chat widget adjusts what it emphasizes based on which visitor role is
selected, and every completed turn is captured with logging that supports
later evaluation of answer quality.

## More than a wrapper around a model call
link: index.html
The chat feature embeds site content into a prebuilt vector index and
coordinates retrieval, gating, and generation around it, going well beyond
a single call out to a language model.

## Applying ML to medical imaging research
link: pages/publications.html
Beyond his game projects, he co-authored a published paper applying a
machine-learning preprocessing method, ps-KDE, to semantic segmentation of
chest X-ray images.

## Six credited academic publications
link: pages/publications.html
The publications page credits him on six separate entries: journal
articles in NeuroImage, Cerebral Cortex, and PLOS ONE, a conference
abstract in Physiology, a PsyArXiv preprint, and conference proceedings
from FBB 2020.

## Core-engine work on the AD toolbox team
link: pages/automatic-differentiation.html
On the automatic differentiation toolbox, a five-person project, he
implemented dual-number arithmetic for the forward-mode engine and led the
correctness testing as QA lead.

## Applied Bayesian coursework at Harvard
link: pages/education.html
His Harvard Medical School biomedical informatics master's (Aug 2021 –
Dec 2022) included coursework in Applied Bayesian Analysis alongside
Machine Learning, beyond purely programming-focused classes.

## Hands-on brain-imaging research methodology
link: pages/publications.html
One of his earlier papers reports a pharmacological fMRI study on how
oxytocin affects self/other distinction in the brain, methodology built on
running actual imaging experiments rather than only reviewing literature.

## A short jam-timeline programming project
link: pages/nothing-can-go-wrong.html
On Nothing Can Go Wrong he was the primary programmer, implementing core
gameplay systems and player control logic in Godot within a short
game-jam development window.

## A CAD replica of a game weapon
link: pages/aegis-sword.html
He modeled a detailed CAD replica of the Aegis Sword from Xenoblade
Chronicles 2 in Onshape, as his final project for a solid-modeling course
at the University of Rochester.

## Where his solo puzzle game is playable
link: pages/gyrotris.html
Gyrotris, the puzzle game he built solo, is published and playable on
itch.io.
