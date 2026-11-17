const PERSONAS = [
  { id: 'none', label: 'General' },
  { id: 'friends', label: 'Friends' },
  { id: 'family', label: 'Family' },
  { id: 'colleagues', label: 'Colleagues' },
  { id: 'experts', label: 'Experts' },
  { id: 'student_teacher', label: 'Student / Teacher' },
]

export default function PersonaSelect({ value, onChange }) {
  return (
    <div className="persona">
      <label className="persona-label" htmlFor="persona-select">
        Writing to
      </label>
      <select
        id="persona-select"
        className="persona-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {PERSONAS.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}
          </option>
        ))}
      </select>
    </div>
  )
}
