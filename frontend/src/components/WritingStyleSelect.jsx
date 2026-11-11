import { WRITING_STYLES } from '../utils/writingStyles'

export default function WritingStyleSelect({ value, onChange }) {
  return (
    <div className="persona">
      <label className="persona-label" htmlFor="style-select">
        Writing style
      </label>
      <select
        id="style-select"
        className="persona-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {WRITING_STYLES.map((s) => (
          <option key={s.id} value={s.id}>
            {s.label}
          </option>
        ))}
      </select>
    </div>
  )
}
