// Writing styles shown in the UI, and the model persona each one maps to.
// The Colab models only understand the persona values (none, friends, family,
// colleagues, experts, student_teacher), so change `persona` here to re-map.
export const WRITING_STYLES = [
  { id: 'formal', label: 'Formal', persona: 'experts' },
  { id: 'semiformal', label: 'Semiformal', persona: 'colleagues' },
  { id: 'informal', label: 'Informal', persona: 'friends' },
]

export const DEFAULT_STYLE = 'informal'

export function isValidStyle(id) {
  return WRITING_STYLES.some((s) => s.id === id)
}

export function styleToPersona(id) {
  const style = WRITING_STYLES.find((s) => s.id === id) || WRITING_STYLES.find((s) => s.id === DEFAULT_STYLE)
  return style.persona
}
