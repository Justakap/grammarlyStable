// Personas understood by the Colab models (correct + predict).
export const PERSONAS = ['none', 'friends', 'family', 'colleagues', 'experts', 'student_teacher']

// Returns a valid persona string, or null if the value is invalid.
// Missing/empty values fall back to "none".
export function parsePersona(value) {
  if (value === undefined || value === null || value === '') return 'none'
  if (typeof value !== 'string') return null
  const normalized = value.trim().toLowerCase().replace(/[\s-]+/g, '_')
  return PERSONAS.includes(normalized) ? normalized : null
}

export const PERSONA_ERROR = `If provided, "persona" must be one of: ${PERSONAS.join(', ')}.`
