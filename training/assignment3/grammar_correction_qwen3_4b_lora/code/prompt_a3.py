def clean_persona(persona):
    if isinstance(persona, str) and persona.strip():
        return persona.strip().lower()
    return "none"


def build_prompt(incorrect, persona=None, use_persona=True):
    if use_persona:
        return (
            "Fix the grammar in this Hinglish sentence.\n"
            f"Persona: {clean_persona(persona)}\n"
            f"Incorrect: {incorrect}\n"
            "Correct: "
        )
    return f"Fix the grammar in this Hinglish sentence.\nIncorrect: {incorrect}\nCorrect: "
