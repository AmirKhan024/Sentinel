import type { ReactNode } from 'react'

export function PageShell({
  title,
  description,
  children,
}: {
  /** Omit when the page renders its own heading (e.g. Overview's hero section). */
  title?: string
  description?: string
  children: ReactNode
}) {
  return (
    <main className="page-shell">
      {title && <h1>{title}</h1>}
      {description && <p className="page-description">{description}</p>}
      {children}
    </main>
  )
}
