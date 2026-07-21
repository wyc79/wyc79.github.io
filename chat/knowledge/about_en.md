# knowledge/about_en.md — English curated chunks (retrieval + en gate)

Sections below are indexed alongside the site pages (see loader.load_knowledge).
They exist because visitors ask in hiring vocabulary the pages themselves never
use — resume, CV, background, qualifications. Keep every fact consistent with
the actual site pages; rebuild the index after editing.

Naming: both languages live in this one folder split by suffix — about_en.md
(this file, English) and about_zh.md (Chinese, the bge-zh gate corpus).
load_knowledge(dir, lang) reads only *_<lang>.md, so the en gate never ingests
zh chunks and vice-versa.

## Resume highlights
link: pages/projects.html
Resume / CV highlights of Yuanchen (YC) Wang: game developer pursuing an M.S.
in Computer Science (Game Development) at the University of Southern
California (2025–2027). Hands-on experience across game design and
engineering: combat design and level design on Cemented Dreams (third-person
platformer — grapple traversal, combat systems, Hive level), engine
programming in C++ on Prime Engine (view frustum culling, physics collision
and sliding response, debug visualization), gameplay design and programming on
Nothing Can Go Wrong, Code Breaker, and Gyrotris. Toolset: Unreal Engine 5,
Unity, Godot, C++, C#, Python. Career background before games: research in
neuroscience, psychology, and biomedical informatics with peer-reviewed
publications. Contact and links: email, LinkedIn, and GitHub are in the site
header.

## Education and qualifications
link: pages/education.html
Education background and qualifications: M.S. Computer Science (Game
Development), University of Southern California, Aug 2025 – May 2027 (3-D
Graphics & Rendering; Game Engine Development). Master of Biomedical
Informatics, Harvard Medical School, 2021–2022 (Machine Learning; Applied
Bayesian Analysis). B.S. Brain & Cognitive Science and B.A. Psychology with a
CS minor, University of Rochester, 2017–2021 (Data Structures & Algorithms;
Artificial Intelligence).

## AI chat agent project (LLM, RAG, agents)
link: index.html
LLM and AI agent engineering: YC designed and built the AI chat agent on
this website end-to-end. It is a retrieval-augmented generation (RAG)
system: site content is chunked and embedded into a static vector index,
sentence embeddings run in the visitor's browser (MiniLM via
transformers.js / ONNX), retrieval is client-side vector search, and answer
generation uses an LLM served through a Cloudflare Worker. The system
includes role-conditioned prompting, prompt-injection and off-topic
defenses, per-turn logging for evaluation, and a tested Python ingestion
pipeline. It demonstrates practical experience with LLM applications,
embeddings, semantic search, and agent-style system design.

## Machine learning research background
link: pages/publications.html
Machine learning background: ML and applied Bayesian analysis coursework at
Harvard Medical School (biomedical informatics master's), the Automatic
Differentiation toolbox project (Python developer & QA lead), and
peer-reviewed publications applying machine learning to medical imaging
(ps-KDE, semantic segmentation of chest X-ray images, PLOS ONE; NeuroImage
neurocognitive research).

## Work experience and project roles
link: pages/projects.html
Work experience and roles held on projects: Combat Designer & Level Designer
(Cemented Dreams), Game Designer & Programmer (Nothing Can Go Wrong), Game
Developer (Code Breaker), Solo Game Developer (Gyrotris), Engine Programmer
(Prime Engine), Designer & Producer (physical game prototypes), Engineer (3D
Rendering Project), CAD Modeler (Aegis Sword), Python Developer & QA Lead
(Automatic Differentiation Toolbox). Research experience includes published
work in NeuroImage, Physiology, and PLOS ONE (ps-KDE, semantic segmentation of
chest X-ray images).
