"""
Shared utilities for emotion concept vector extraction.

Provides data loading, model configuration, and prompt formatting used
by generate_stories.py, generate_neutral.py, and extract_emotion_vectors.py.
"""

import re
from pathlib import Path


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_emotions(path: str = "emotions/emotions_list.txt") -> list[str]:
    """Parse comma-separated emotions file. Returns list of stripped strings."""
    text = Path(path).read_text().strip()
    emotions = [e.strip() for e in text.split(",") if e.strip()]
    assert len(emotions) == 171, f"Expected 171 emotions, got {len(emotions)}"
    return emotions


def load_topics(path: str = "emotions/story_topics.txt") -> list[str]:
    """Parse blank-line-separated topics file. Returns list of stripped strings."""
    text = Path(path).read_text()
    topics = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(topics) == 100, f"Expected 100 topics, got {len(topics)}"
    return topics


# ─── Model Configuration ────────────────────────────────────────────────────


def get_model_id(model_name: str) -> str:
    """Convert model name to filesystem-safe identifier.

    Examples:
        "Qwen/Qwen3-4B-Instruct-2507" -> "qwen3_4b_instruct"
        "openai/gpt-oss-20b" -> "gpt_oss_20b"
        "Qwen/Qwen3-8B" -> "qwen3_8b"
    """
    # Take the part after the slash (if present)
    name = model_name.split("/")[-1]
    # Remove trailing version numbers like -2507
    name = re.sub(r"-\d{4}$", "", name)
    # Lowercase and replace hyphens with underscores
    name = name.lower().replace("-", "_")
    return name


def get_artifact_dir(model_name: str) -> Path:
    """Return the artifact directory for a given model."""
    return Path("emotions/artifacts") / get_model_id(model_name)


# ─── Prompt Formatting ──────────────────────────────────────────────────────


STORY_PROMPT_TEMPLATE = """\
Write a short story based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

The story should be roughly one paragraph. Use either third-person narration or first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


NEUTRAL_PROMPT_TEMPLATE = """\
Write a dialogue based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogue like so:

[optional system instructions]

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[continue for 2-6 exchanges]

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

The dialogue should be one of these types:
- Code or programming tasks
- Factual questions (science, history, math, geography)
- Work-related tasks (writing, analysis, summarization)
- Practical how-to questions
- Creative but neutral tasks (brainstorming names, generating lists)

Optionally include a system prompt at the start, before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.

CRITICAL REQUIREMENT: This dialogue must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


def format_story_prompt(emotion: str, topic: str) -> str:
    """Return the story generation prompt for a given emotion and topic."""
    return STORY_PROMPT_TEMPLATE.format(emotion=emotion, topic=topic)


def format_neutral_prompt(topic: str) -> str:
    """Return the neutral dialogue generation prompt for a given topic."""
    return NEUTRAL_PROMPT_TEMPLATE.format(topic=topic)
