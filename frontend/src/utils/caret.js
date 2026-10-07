/**
 * Character-offset-based caret helpers for a contentEditable element.
 * React can't "control" contentEditable the way it controls an <input>,
 * so the editor manages its DOM imperatively and uses these to save/restore
 * the caret around those DOM rewrites.
 */

export function getCaretOffset(root) {
  const selection = window.getSelection()
  if (!root || !selection || selection.rangeCount === 0) return null

  const range = selection.getRangeAt(0)
  if (!root.contains(range.startContainer)) return null

  const preRange = document.createRange()
  preRange.selectNodeContents(root)
  preRange.setEnd(range.startContainer, range.startOffset)
  return preRange.toString().length
}

export function setCaretOffset(root, offset) {
  if (!root) return
  const selection = window.getSelection()
  if (!selection) return

  let remaining = offset
  let targetNode = null
  let targetOffset = 0

  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const len = node.textContent.length
      if (remaining <= len) {
        targetNode = node
        targetOffset = remaining
        return true
      }
      remaining -= len
      return false
    }
    for (const child of node.childNodes) {
      if (walk(child)) return true
    }
    return false
  }

  walk(root)

  const range = document.createRange()
  if (targetNode) {
    range.setStart(targetNode, targetOffset)
  } else {
    range.selectNodeContents(root)
    range.collapse(false)
  }
  range.collapse(true)
  selection.removeAllRanges()
  selection.addRange(range)
}
